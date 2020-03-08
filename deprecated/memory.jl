using Boilerplate
using JsonBuilder

Boilerplate.setup_repl().web_display(echarts=true).load_std().load_snippets()

a = readlines("log")
a = filter(x->startswith(x, "  -> memory"), a)
a = map(a) do line
    line = split(line)
    gpu = parse(Int, line[3])
    t = parse(Int, line[4]) / 1000
    m = parse(Int, line[6]) / 1000_000
    gpu, t, m
end

# b = readlines("memory.txt")
# b = b[findfirst(x->!endswith(x, " 0 0 0 0"), b):findlast(x->!endswith(x, " 0 0 0 0"), b)]
# b = b[1850:end] # sucks
# b = b[findfirst(x->!endswith(x, " 0 0 0 0"), b):findlast(x->!endswith(x, " 0 0 0 0"), b)]
# start = parse(f64, split(car(b)) |> car)
# b = map(b) do line
#     line = split(line)
#     t = round(Int, parse(f64, line[1]) - start) / 20
#     m1 = parse(Int, line[3])
#     m2 = parse(Int, line[5])
#     t, m1, m2
# end

c0 = readlines("g0")[2:end]
c0 = map(c0) do line
    line = split(line)
    t = parse(Int, line[1]) / 1100
    m = parse(Int, line[2]) / 1000_000
    (t, m)
end

c1 = readlines("g1")[2:end]
c1 = map(c1) do line
    line = split(line)
    t = parse(Int, line[1]) / 1100
    m = parse(Int, line[2]) / 1000_000
    (t, m)
end

plot(@json """{
    backgroundColor: 'transparent',
    title: { text: 'memory' },
    xAxis: { type: 'value' },
    yAxis: { type: 'value' },
    legend: { width: 400 },
    tooltip: { trigger: 'item', formatter: '{a} {c}' },
    series: [{
        name: 'g0_sim',
        type: 'line',
        data: [$([(t, m) for (gpu, t, m) in a if gpu == 0])...]
    }, {
        name: 'g1_sim',
        type: 'line',
        data: [$([(t, m) for (gpu, t, m) in a if gpu == 1])...]
    }, {
        name: 'g0_meta',
        type: 'line',
        data: [$c0...]
    }, {
        name: 'g1_meta',
        type: 'line',
        data: [$c1...]
    }]
}""")

"""
, {
        name: 'g0_nv',
        type: 'line',
        data: [$([(t, m1) for (t, m1, m2) in b])...]
    }, {
        name: 'g1_nv',
        type: 'line',
        data: [$([(t, m2) for (t, m1, m2) in b])...]
    },
"""
