import os, shutil

if __name__ ==  "__main__":
    mode = 'train'
    inPath = '/home/lpc/dataset/VSR/vimeo_triplet_small/sequences/'
    outPath = '/home/lpc/dataset/VSR/vimeo_triplet_small/sequences/' + mode + '/'
    guide = '/home/lpc/dataset/VSR/vimeo_triplet_small/tri_' + mode + 'list.txt'
    
    f = open(guide, "r")
    lines = f.readlines()
    
    if not os.path.isdir(outPath):
        os.mkdir(outPath)

    for l in lines:
        line = l.replace('\n','')
        this_folder = os.path.join(inPath, line)
        dest_folder = os.path.join(outPath, line)
        print(this_folder)
        shutil.copytree(this_folder, dest_folder)
    print('Done')
