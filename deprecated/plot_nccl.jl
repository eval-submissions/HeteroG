using Boilerplate
using JsonBuilder

Boilerplate.setup_repl().web_display(echarts=true).load_std().load_snippets()

WebDisplay.extra_header[] = """
    <script src="https://gdrive.ylxdzsw.com/web-assets/echarts.min.js"></script>
    <script src="https://gdrive.ylxdzsw.com/web-assets/dark.js"></script>
"""

intra = [(1, 218), (2, 257), (4, 206), (8, 202), (16, 291), (32, 265), (64, 345), (128, 284), (256, 337), (512, 418), (1024, 518), (2048, 713), (4096, 1119), (8192, 2008), (16384, 3773), (32768, 7206), (65536, 14039), (131072, 27725), (262144, 55431), (524288, 110882), (1048576, 221519)]
inter = [(1, 335), (2, 334), (4, 263), (8, 354), (16, 282), (32, 320), (64, 269), (128, 287), (256, 404), (512, 518), (1024, 789), (2048, 1188), (4096, 1359), (8192, 2581), (16384, 4788), (32768, 9267), (65536, 18460), (131072, 36235), (262144, 72982), (524288, 145867), (1048576, 291205)]

plot(@json """{
    backgroundColor: 'transparent',
    title: { text: 'intra task', left: 'center' },
    xAxis: { name: 'log(KB)', type: 'value' },
    yAxis: { name: 'log(us)', type: 'value' },
    legend: { width: 400, top: 28 },
    tooltip: { trigger: 'item', formatter: '{a} {c}' },
    series: [{
        name: 'real',
        type: 'scatter',
        data: $([(log2(x), log2(y)) for (x, y) in intra])
    }, {
        name: 'model',
        type: 'line',
        data: [[0, 8.2], [10, 8.2], [20, 18]]
    }]
}""")

plot(@json """{
    backgroundColor: 'transparent',
    title: { text: 'inter task', left: 'center' },
    xAxis: { name: 'log(KB)', type: 'value' },
    yAxis: { name: 'log(us)', type: 'value' },
    legend: { width: 400, top: 28 },
    tooltip: { trigger: 'item', formatter: '{a} {c}' },
    series: [{
        name: 'real',
        type: 'scatter',
        data: $([(log2(x), log2(y)) for (x, y) in inter])
    }, {
        name: 'model',
        type: 'line',
        data: [[0, 8.2], [10, 8.2], [20, 18]]
    }]
}""")