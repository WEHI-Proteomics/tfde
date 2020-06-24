import Foundation
import SQLite
import SwiftGD

#if os(Linux)
let CONVERTED_DATABASE_NAME = "/data/exp-dwm-test-run-190719_Hela_Ecoli_1to1_01-converted.sqlite"
#else
let CONVERTED_DATABASE_NAME = "/Users/darylwilding-mcbride/Downloads/experiments/dwm-test/converted-databases/exp-dwm-test-run-190719_Hela_Ecoli_1to1_01-converted.sqlite"
#endif
let db = try Connection(CONVERTED_DATABASE_NAME)


let PIXELS_X = 910
let PIXELS_Y = 910  // equal to the number of scan lines
let MZ_MIN = 100.0
let MZ_MAX = 1700.0
let SCAN_MAX = PIXELS_Y
let SCAN_MIN = 1
let MZ_PER_TILE = 18.0
let TILES_PER_FRAME = Int((MZ_MAX - MZ_MIN) / MZ_PER_TILE) + 1
let MIN_TILE_IDX = 0
let MAX_TILE_IDX = TILES_PER_FRAME-1

let MINIMUM_PIXEL_INTENSITY = 1
let MAXIMUM_PIXEL_INTENSITY = 1000


func greet(person: String) -> String {
    let greeting = "Hello, " + person + "!"
    return greeting
}

func tileAndPixelXFromMz(Double: mz) -> (Int, Int) {
    let mz_adj = mz - MZ_MIN
    let tile_id = Int(mz_adj / MZ_PER_TILE)
    let pixel_x = Int((mz_adj % MZ_PER_TILE) / MZ_PER_TILE * PIXELS_X)
    return (tile_id, pixel_x)
}

// figure out where to save our file
let currentDirectory = URL(fileURLWithPath: FileManager().currentDirectoryPath)
let destination = currentDirectory.appendingPathComponent("output-1.png")

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
print("reading the database")
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
