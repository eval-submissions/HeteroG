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
a = [[1, 101],
[2, 186],
[3, 268],
[4, 354],
[5, 443],
[6, 524],
[7, 615],
[8, 696],
[10, 870],
[12, 1042],
[14, 1216],
[16, 1387],
[20, 1733],
[24, 2079],
[28, 2425],
[32, 2771],
[40, 3462],
[48, 4153]]

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
        data: [
            #[0,-23.27],
            [0,9.19],
            #[48,17387.29]]
            [48, 4150.8]
        ]
    }, {
        type: 'line',
        name: 'eastimated',
        data: [
            #[0,1.03],
            [0, 1.50],
            #[48,8980.91]
            [48, 1944.90]
        ]
    }]
}""")

isinteractive() || wait()
