import Foundation
import SQLite
import SwiftGD

#if os(Linux)
let CONVERTED_DATABASE_NAME = "/data/exp-dwm-test-run-190719_Hela_Ecoli_1to1_01-converted.sqlite"
let COLOURMAP_DATABASE_NAME = "/data/colourmap.sqlite"
let TILE_BASE_DIR = "/data/tiles"
#else
let CONVERTED_DATABASE_NAME = "/Users/darylwilding-mcbride/Downloads/experiments/dwm-test/converted-databases/exp-dwm-test-run-190719_Hela_Ecoli_1to1_01-converted.sqlite"
let COLOURMAP_DATABASE_NAME = "/Users/darylwilding-mcbride/Downloads/colourmap.sqlite"
let TILE_BASE_DIR = "/Users/darylwilding-mcbride/Downloads/swift-tiles"
#endif


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

// frame types for PASEF mode
let FRAME_TYPE_MS1 = 0
let FRAME_TYPE_MS2 = 8

let clippedColour = Color(red: 1.0, green: 0.0, blue: 0.0, alpha: 1.0)

// this function bins the m/z value to fit into a pixel
func tileAndPixelXFromMz(mz: Double) -> (Int64, Int64) {
    let mzAdj = mz - MZ_MIN
    let tileId = Int64(mzAdj / MZ_PER_TILE)
    let pixelX = Int64((mzAdj.truncatingRemainder(dividingBy: MZ_PER_TILE)) / MZ_PER_TILE * Double(PIXELS_X))
    return (tileId, pixelX)
}

func colourMappingForIntensity(intensity: Int64, colourLookup: [IntensityColourMapping]) -> IntensityColourMapping {
    // look up the colour for this intensity
    var intensityColourMapping: IntensityColourMapping
    if intensity <= MAXIMUM_PIXEL_INTENSITY {
        intensityColourMapping = colourLookup[Int(intensity)-1]
    }
    else {
        intensityColourMapping = colourLookup.last ?? IntensityColourMapping(intensity: Int64(intensity), colour: clippedColour)
    }
    return intensityColourMapping
}

struct GroupedTilePixel {
    var tileId: Int64
    var pixelX: Int64
    var scan: Int64
    var intensity: Int64
    var colour: Color
}

struct IntensityColourMapping {
    var intensity: Int64
    var colour: Color
}

extension Collection {
    func choose(_ n: Int) -> ArraySlice<Element> { shuffled().prefix(n) }
}

// frames table
let frames = Table("frames")
let frameId = Expression<Int>("frame_id")
let mz = Expression<Double>("mz")
let scan = Expression<Int64>("scan")
let intensity = Expression<Int64>("intensity")

// colourmap table
let colourmapping = Table("colours")
let cmIntensity = Expression<Int64>("intensity")
let red = Expression<Double>("r")
let green = Expression<Double>("g")
let blue = Expression<Double>("b")

// frame_properties table
let frameProperties = Table("frame_properties")
let framePropertiesId = Expression<Int>("Id")
let frameTime = Expression<Double>("Time")
let frameType = Expression<Int>("MsMsType")

// load the mapping of intensity to colour
let colourmap_db = try Connection(COLOURMAP_DATABASE_NAME)
var colourLookup: [IntensityColourMapping] = []
for row in try colourmap_db.prepare(colourmapping) {
    colourLookup.append(IntensityColourMapping(intensity: row[cmIntensity], colour: Color(red: row[red], green: row[green], blue: row[blue], alpha: 1.0)))
}


// load the MS1 frame IDs
print("load the frame ids")
let db = try Connection(CONVERTED_DATABASE_NAME)
var frameIDs: [Int] = []
for frameProperty in try db.prepare(frameProperties.filter(frameType == FRAME_TYPE_MS1)) {
    frameIDs.append(Int(frameProperty[framePropertiesId]))
}
let ms1FrameIDs = frameIDs.choose(50)


// generate the tiles for each frame
print("render the frames")
var elapsedTimes: [Double] = []
var tileIds = Set<Int64>()
for ms1FrameId in ms1FrameIDs {
    let startTime = ProcessInfo.processInfo.systemUptime

    var groupedPixels: [String:GroupedTilePixel] = [:]
    for point in try db.prepare(frames.filter(frameId == ms1FrameId)) {
        let (tileId, pixelX) = tileAndPixelXFromMz(mz: point[mz])
        tileIds.insert(tileId)
        let key = "\(tileId)-\(pixelX)-\(point[scan])"
        var groupedPixel:GroupedTilePixel
        // check if the key for this grouped pixel is already there; update its intensity if so; add it if not
        if groupedPixels.keys.contains(key) {
            groupedPixel = groupedPixels[key]!
            groupedPixel.intensity += point[intensity]
        }
        else {
            groupedPixel = GroupedTilePixel(tileId: tileId, pixelX: pixelX, scan: point[scan], intensity: Int64(point[intensity]), colour: Color(red: 0.0, green: 0.0, blue: 0.0, alpha: 1.0))
        }
        groupedPixels[key] = groupedPixel
    }
    let groupedPixelsArr = Array(groupedPixels.values.map{ $0 })


    // generate the tiles for this frame
    for tileId in tileIds {
        let tilePath = URL(fileURLWithPath: "\(TILE_BASE_DIR)/frame-\(ms1FrameId)-tile-\(tileId).png")

        let filtered = groupedPixelsArr.filter{ $0.tileId == tileId } 
        if let image = Image(width: PIXELS_X, height: PIXELS_Y) {
            // set the pixels in the image
            for pixel in filtered {
                // look up the colour for this intensity
                let intensityColourMapping = colourMappingForIntensity(intensity: pixel.intensity, colourLookup: colourLookup)
                // set the pixel
                image.set(pixel: Point(x: Int(pixel.pixelX), y: Int(pixel.scan)), to: intensityColourMapping.colour)
            }

            // save the final image to disk
            image.write(to: tilePath, allowOverwrite:true)
        }
    }
    let timeElapsed = ProcessInfo.processInfo.systemUptime - startTime
    elapsedTimes.append(timeElapsed)
    let avgTime = Float(elapsedTimes.reduce(0, +)) / Float(elapsedTimes.count)
    print("processed frame \(ms1FrameId) in \(timeElapsed) secs - average \(avgTime)")
}

print("finished")
