mkdir pack
cp model.pb modified.pb timeline.json simulated.json meta.pb pack
tar -czvf pack.tar.gz pack
rm -rf pack
