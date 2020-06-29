import Foundation
import SQLite
import SwiftGD
import SwiftyJSON

#if os(Linux)
let CONVERTED_DATABASE_NAME = "/data/exp-dwm-test-run-190719_Hela_Ecoli_1to1_01-converted.sqlite"
let TILE_BASE_DIR = "/data/tiles"
#else
let CONVERTED_DATABASE_NAME = "/Users/darylwilding-mcbride/Downloads/experiments/dwm-test/converted-databases/exp-dwm-test-run-190719_Hela_Ecoli_1to1_01-converted.sqlite"
let TILE_BASE_DIR = "/Users/darylwilding-mcbride/Downloads/swift-tiles"
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

let MINIMUM_PIXEL_INTENSITY: Int64 = 1
let MAXIMUM_PIXEL_INTENSITY: Int64 = 1000


func greet(person: String) -> String {
    let greeting = "Hello, " + person + "!"
    return greeting
}

func tileAndPixelXFromMz(mz: Double) -> (Int64, Int64) {
    let mzAdj = mz - MZ_MIN
    let tileId = Int64(mzAdj / MZ_PER_TILE)
    let pixelX = Int64((mzAdj.truncatingRemainder(dividingBy: MZ_PER_TILE)) / MZ_PER_TILE * Double(PIXELS_X))
    return (tileId, pixelX)
}

struct PixelColour {
    var red: Double
    var green: Double
    var blue: Double
}

struct TilePixel {
    var mz: Double
    var scan: Int64
    var intensity: Int64
    var tileId: Int64
    var pixelX: Int64
}

struct GroupedTilePixel {
    var tileId: Int64
    var pixelX: Int64
    var scan: Int64
    var intensity: Int64
    var colour: PixelColour
}

struct IntensityColourMapping {
    var intensity: Int64
    var colour: PixelColour
}

let clipped = PixelColour(red: 1.0, green: 0.0, blue:0.0)

let frames = Table("frames")
let frameId = Expression<Int64>("frame_id")
let mz = Expression<Double>("mz")
let scan = Expression<Int64>("scan")
let intensity = Expression<Int64>("intensity")

// load colour table
let path = URL(fileURLWithPath: "/Users/darylwilding-mcbride/Downloads/colourmap.json")
let data = try Data(contentsOf: path, options: .mappedIfSafe)
let json = try JSON(data: data)

var colourLookup: [IntensityColourMapping] = []
for (_, subJson) in json {
    let colour: JSON = subJson["colour"]
    let r: Double = colour[0].double ?? 0.0 / 255.0
    let g: Double = colour[1].double ?? 0.0 / 255.0
    let b: Double = colour[2].double ?? 0.0 / 255.0

    let intensity: Int64 = subJson["intensity"].int64 ?? 0

    let intensityColour = IntensityColourMapping(intensity: intensity, colour: PixelColour(red: r, green: g, blue: b))
    colourLookup.append(intensityColour)
}

// build the array of tile pixels
print("reading the database")
var tilePixels: [TilePixel] = []
for frame in try db.prepare(frames.filter(frameId == 1899)) {
    let (tileId, pixelX) = tileAndPixelXFromMz(mz: frame[mz])
    let pixel = TilePixel(mz: frame[mz], scan: frame[scan], intensity: frame[intensity], tileId:tileId, pixelX: pixelX)
    tilePixels.append(pixel)
}


// sum the intensity values in the same (tile,scan,pixelX)
let dict = Dictionary(grouping: tilePixels) { [$0.tileId, $0.pixelX, $0.scan] }
var groupedTilePixels: [GroupedTilePixel] = []
for (key,value) in dict{
    // calculate the total intensity for this pixel
    var totalIntensity = 0
    for o in value {
        totalIntensity += Int(o.intensity)
    }

    // look up the colour for this intensity
    var intensityColourMapping: IntensityColourMapping = IntensityColourMapping(intensity: MAXIMUM_PIXEL_INTENSITY, colour: clipped)
    if totalIntensity <= MAXIMUM_PIXEL_INTENSITY {
        intensityColourMapping = colourLookup[totalIntensity-1]
    }
    let colour = intensityColourMapping.colour
    let pixelColour = PixelColour(red: colour.red, green: colour.green, blue: colour.blue)

    groupedTilePixels.append(GroupedTilePixel(tileId: key[0], pixelX: key[1], scan: key[2], intensity: Int64(totalIntensity), colour: pixelColour))
}

// generate the tiles for this frame
for tileId in MIN_TILE_IDX...MAX_TILE_IDX {
    let tilePath = URL(fileURLWithPath: "\(TILE_BASE_DIR)/tile-\(tileId).png")

    let filtered = groupedTilePixels.filter{ $0.tileId == tileId } 
    if let image = Image(width: PIXELS_X, height: PIXELS_Y) {
        // set the pixels in the image
        for pixel in filtered {
            let c = Color(red: pixel.colour.red, green: pixel.colour.green, blue: pixel.colour.blue, alpha: 1)
            image.set(pixel: Point(x: Int(pixel.pixelX), y: Int(pixel.scan)), to: c)
        }

        // save the final image to disk
        image.write(to: tilePath)
    }
}

print("finished")
