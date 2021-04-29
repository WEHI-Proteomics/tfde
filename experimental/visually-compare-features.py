import os
import shutil
import glob
from os.path import expanduser

# Note this script uses the convert command from ImageMagick, installed with:
# sudo apt install imagemagick


BASE_DIR = expanduser('~')
OVERLAY_A_BASE_DIR = '{}/precursor-cuboid-tiles'.format(BASE_DIR)
OVERLAY_B_BASE_DIR = '{}/precursor-cuboid-3did-tiles'.format(BASE_DIR)

overlay_A_files_l = sorted(glob.glob('{}/*.png'.format(OVERLAY_A_BASE_DIR)), key=lambda x: ( int(x.split('tile-')[1].split('.png')[0]) ))
overlay_B_files_l = sorted(glob.glob('{}/*.png'.format(OVERLAY_B_BASE_DIR)), key=lambda x: ( int(x.split('tile-')[1].split('.png')[0]) ))


# check the composite tiles directory - the composites will be put in the tile list A directory
COMPOSITE_TILE_BASE_DIR = '{}/composite-tiles'.format(BASE_DIR)
if os.path.exists(COMPOSITE_TILE_BASE_DIR):
    shutil.rmtree(COMPOSITE_TILE_BASE_DIR)
os.makedirs(COMPOSITE_TILE_BASE_DIR)

# for each tile in the tile list, find its A and B overlay, and create a composite of them
composite_tile_count = 0
for idx,f in enumerate(overlay_A_files_l):
    overlay_a_name = f
    overlay_b_name = overlay_B_files_l[idx]
    print('compositing {} and {} as tile {}'.format(overlay_a_name, overlay_b_name, idx+1))

    composite_name = '{}/composite-tile-{}.png'.format(COMPOSITE_TILE_BASE_DIR, idx+1)

    # make the composite
    if os.path.isfile(overlay_a_name) and os.path.isfile(overlay_b_name):
        cmd = "convert {} {} +append -background darkgrey -splice 10x0+800+0 {}".format(overlay_a_name, overlay_b_name, composite_name)
        os.system(cmd)
        composite_tile_count += 1
    else:
        if not os.path.isfile(overlay_a_name):
            print('could not find {}'.format(overlay_a_name))
        if not os.path.isfile(overlay_b_name):
            print('could not find {}'.format(overlay_b_name))
    # print('.', end='', flush=True)

print()
print('wrote {} composite tiles to {}'.format(composite_tile_count, COMPOSITE_TILE_BASE_DIR))
