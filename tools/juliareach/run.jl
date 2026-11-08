# JuliaReach (ReachabilityAnalysis.jl, TMJets) on one benchmark.
#
#   julia --project=tools/juliareach tools/juliareach/run.jl <config.json> <settings.json> <out.json>
#
# Called by tools/juliareach/run.py, which turns the output into a result record.
# The benchmark is a forward problem from one initial box to one target box;
# run.py reports every other benchmark as unsupported. The vector field is
# generated from the config's equation strings. Every flowpipe segment is
# enclosed in a box and compared with the target box:
#   disjoint  the segment's states are certainly outside the target
#   inside    the segment's states are certainly inside the target
#   possible  otherwise
# A segment inside the target proves that the target is reached (the segments
# are over-approximations, so every trajectory is then in the target); no
# intersection at all proves that it is not reached; an intersection alone
# proves nothing.

using ReachabilityAnalysis, LazySets, JSON3
using Pkg

const cfg = JSON3.read(read(ARGS[1], String))
const st = JSON3.read(read(ARGS[2], String))
const out_path = ARGS[3]

const n = length(cfg.system.equations)

julia_expr(s) = replace(replace(String(s), "**" => "^"), r"\bx(\d+)\b" => s"x[\1]")

# --- vector field -----------------------------------------------------------
let lines = ["dx[$i] = (" * julia_expr(e) * ")" for (i, e) in enumerate(cfg.system.equations)]
    body = join(lines, "\n    ")
    macro_ = get(st, :taylorize, true) ? "@taylorize " : ""
    eval(Meta.parse("""begin
    $(macro_)function field!(dx, x, p, t)
        $body
        return dx
    end
    end"""))
end

# --- sets -------------------------------------------------------------------
# the one box of a list of sets
function only_box(sets)
    (length(sets) == 1 && sets[1].type == "box") || error("one box is expected")
    return Hyperrectangle(low=Float64[b[1] for b in sets[1].bounds], high=Float64[b[2] for b in sets[1].bounds])
end

# --- analysis ---------------------------------------------------------------
const t0, t1 = Float64(cfg.time_horizon[1]), Float64(cfg.time_horizon[2])
const X0 = only_box(cfg.initial_sets)
const target = only_box(cfg.target_sets)
const alg = TMJets(abstol=st.abstol, orderT=st.orderT, orderQ=st.orderQ, maxsteps=st.maxsteps)

function solve_from(X0, T)
    prob = @ivp(x' = field!(x), dim: n, x(0) ∈ X0)
    return solve(prob, T=T, alg=alg)
end

# warm-up on a tiny horizon so that compilation is not part of the timed solve
warmup = @elapsed Base.invokelatest(solve_from, X0, min(t1, 1e-3))

segments = Tuple{Float64,Float64}[]
flowpipe = Vector{Vector{Float64}}()  # per segment: t0, t1, box lower bounds, upper bounds, status (0 disjoint, 1 possible, 2 inside)
num_possible = 0
num_inside = 0
sol = nothing
solve_time = @elapsed (sol = Base.invokelatest(solve_from, X0, t1))
reached_until = tend(sol)
covered = reached_until >= t1 - 1e-9
check_time = @elapsed for R in sol
    global num_possible, num_inside, reached_until, covered
    a, b = tstart(R), tend(R)
    b < t0 && continue
    B = try
        set(overapproximate(R, Hyperrectangle))
    catch err  # the Taylor model has no finite enclosure any more (NaN remainder): the flowpipe ends here
        err isa AssertionError || rethrow()
        nothing
    end
    if B === nothing || !all(isfinite, low(B)) || !all(isfinite, high(B))
        reached_until = a
        covered = false
        break
    end
    status = isdisjoint(B, target) ? 0 : (issubset(B, target) ? 2 : 1)
    push!(flowpipe, [a; b; low(B); high(B); Float64(status)])
    num_possible += status == 1
    num_inside += status == 2
    status > 0 && push!(segments, (max(a, t0), min(b, t1)))
end

verdict = if num_inside > 0
    "reachable_proven"
elseif !covered
    "incomplete"
elseif num_possible == 0
    "unreachable_proven"
else
    "reachable_possible"
end

function merged(iv)
    out = Tuple{Float64,Float64}[]
    for (a, b) in sort(iv)
        if !isempty(out) && a <= out[end][2]
            out[end] = (out[end][1], max(out[end][2], b))
        else
            push!(out, (a, b))
        end
    end
    out
end

versions = Dict(info.name => string(info.version) for (_, info) in Pkg.dependencies()
                if info.is_direct_dep && info.version !== nothing)
versions["julia"] = string(VERSION)

result = Dict(
    "verdict" => verdict,
    "horizon_covered" => covered,
    "flowpipe_end_time" => reached_until,
    "num_segments" => length(flowpipe),
    "num_possible" => num_possible,
    "num_inside" => num_inside,
    "reach_time_bounds" => [[a, b] for (a, b) in merged(segments)],
    "times" => Dict("warmup_compile" => warmup, "solve" => solve_time, "check" => check_time),
    "versions" => versions,
)
open(out_path, "w") do io
    JSON3.write(io, result)
end
# the segment boxes behind the verdict, for plotting (scripts/report.py)
open(joinpath(dirname(out_path), "flowpipe.json"), "w") do io
    JSON3.write(io, Dict("n" => n, "status_codes" => ["disjoint", "possible", "inside"],
                         "segments" => [round.(r; sigdigits=6) for r in flowpipe]))
end
