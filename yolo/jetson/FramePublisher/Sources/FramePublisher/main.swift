import SQLite
import PythonKit
import Foundation
import SwiftGD

// let CONVERTED_DATABASE_NAME = "/data/exp-dwm-test-run-190719_Hela_Ecoli_1to1_01-converted.sqlite"
let CONVERTED_DATABASE_NAME = "/Users/darylwilding-mcbride/Downloads/experiments/dwm-test/converted-databases/exp-dwm-test-run-190719_Hela_Ecoli_1to1_01-converted.sqlite"
let PUBLISHED_FRAMES_DIR = "/data/published-frames"

let db = try Connection(CONVERTED_DATABASE_NAME)

print("reading the database")

let query = "select mz,scan,intensity from frames where frame_id == 1899 limit 1000"

// figure out where to save our file
let currentDirectory = URL(fileURLWithPath: FileManager().currentDirectoryPath)
let destination = currentDirectory.appendingPathComponent("output-1.png")

let PIXELS_X = 910
let PIXELS_Y = 910

// if let image = Image(width: PIXELS_X, height: PIXELS_Y) {
//     // for each 
//     for p in try db.prepare(query) {
//         print("m/z: \(p[0] ?? 0.0), scan: \(p[1] ?? 0.0), intensity: \(p[2] ?? 0)")
//     }


//     // image.set(pixel: Point(x: x, y: y), to: Color(red: 0.6, green: 0.8, blue: 0.2, alpha: 1))


//     // save the final image to disk
//     image.write(to: destination)
// }

let list: PythonObject = [0, 1, 2]
let t = PythonObject(tupleOf: 1, 2)

list.append(3)
print("list: \(list)")

var numpyModule = try? Python.attemptImport("numpy")
guard let np = numpyModule else { throw NSError(domain: "could not import numpy", code: 0, userInfo: [:] ) }
// let numpyArrayEmpty = np.array([] as [Float], dtype: np.float32)


// let db_conn = sqlite3.connect(converted_db_name)
// let raw_points_df = pd.read_sql_query("select mz,scan,intensity from frames where frame_id == {}".format(1899), db_conn)
// db_conn.close()


print("finished")
