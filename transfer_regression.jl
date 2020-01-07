using Boilerplate
using JsonBuilder

Boilerplate.setup_repl().web_display(echarts=true).load_std().load_snippets()

# 1080ti
a = [[1,413],
[2,645],
[3,1216],
[4,1609],
[5,2027],
[6,1918],
[7,2827],
[8,2545],
[10,3188],
[12,4821],
[14,4456],
[16,6426],
[20,8032],
[24,7628],
[28,11243],
[32,10168],
[40,12710],
[48,19271]]

# v100 nvlink, raw bandwidth: 9.23 (p2p disable) 24.12 (p2p enable)


X = map(car, a)
X = [X X]
X[:, 2] .= 1
y = map(cadr, a)
X\y

plot(@json """{
    backgroundColor: 'transparent',
    title: { text: 'NV Link' },
    xAxis: { type: 'value' },
    yAxis: { type: 'value' },
    legend: { width: 400 },
    tooltip: { trigger: 'item', formatter: '{a} {c}' },
    series: [{
        name: 'measure',
        type: 'scatter',
        data: [$a...]
    }, {
        type: 'line',
        name: 'regression',
        data: [#[0,-23.27], # this is the real data but it makes the plot ugly
            [0,0],
            [48,17387.29]]
    }, {
        type: 'line',
        name: 'eastimated',
        data: [[0,1.03],
            [48,8980.91]]
    }]
}""")

isinteractive() || wait()
