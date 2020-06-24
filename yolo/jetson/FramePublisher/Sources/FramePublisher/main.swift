import Foundation
import SQLite
import SwiftGD

#if os(Linux)
let CONVERTED_DATABASE_NAME = "/data/exp-dwm-test-run-190719_Hela_Ecoli_1to1_01-converted.sqlite"
#else
let CONVERTED_DATABASE_NAME = "/Users/darylwilding-mcbride/Downloads/experiments/dwm-test/converted-databases/exp-dwm-test-run-190719_Hela_Ecoli_1to1_01-converted.sqlite"
#endif


let db = try Connection(CONVERTED_DATABASE_NAME)

print("reading the database")

let query = "select mz,scan,intensity from frames where frame_id == 1899 limit 1000"

// figure out where to save our file
let currentDirectory = URL(fileURLWithPath: FileManager().currentDirectoryPath)
let destination = currentDirectory.appendingPathComponent("output-1.png")

let PIXELS_X = 910
let PIXELS_Y = 910

struct TilePixel {
    var mz: Double
    var scan: Int64
    var intensity: Int64
    var tileId: Int64
    var pixelX: Int64
    var colourR: Double
    var colourG: Double
    var colourB: Double
}

let frames = Table("frames")
let frameId = Expression<Int64>("frame_id")
let mz = Expression<Double>("mz")
let scan = Expression<Int64>("scan")
let intensity = Expression<Int64>("intensity")


// build the array of tile pixels
var tilePixels: [TilePixel] = []
for frame in try db.prepare(frames.filter(frameId == 1899)) {
    // print("id: \(frame[frameId]), m/z: \(frame[mz]), scan: \(frame[scan])")
    let pixel = TilePixel(mz: frame[mz], scan: frame[scan], intensity: frame[intensity], tileId:0, pixelX: 0, colourR: 0.2, colourG: 0.4, colourB: 0.8)
    tilePixels.append(pixel)
}


// if let image = Image(width: PIXELS_X, height: PIXELS_Y) {
//     // for each 
//     for p in try db.prepare(query) {
//         print("m/z: \(p[0] ?? 0.0), scan: \(p[1] ?? 0.0), intensity: \(p[2] ?? 0)")
//     }


//     // image.set(pixel: Point(x: x, y: y), to: Color(red: 0.6, green: 0.8, blue: 0.2, alpha: 1))


//     // save the final image to disk
//     image.write(to: destination)
// }

print("finished")
