import SQLite

// let CONVERTED_DATABASE_NAME = "/data/exp-dwm-test-run-190719_Hela_Ecoli_1to1_01-converted.sqlite"
let CONVERTED_DATABASE_NAME = "/Users/darylwilding-mcbride/Downloads/experiments/dwm-test/converted-databases/exp-dwm-test-run-190719_Hela_Ecoli_1to1_01-converted.sqlite"
let PUBLISHED_FRAMES_DIR = "/data/published-frames"

let db = try Connection(CONVERTED_DATABASE_NAME)

print("reading the database")

let query = "select mz,scan,intensity from frames where frame_id == 1899 limit 1000"
for p in try db.prepare(query) {
    print("m/z: \(p[0] ?? 0.0), scan: \(p[1] ?? 0.0), intensity: \(p[2] ?? 0)")
}

print("finished")
