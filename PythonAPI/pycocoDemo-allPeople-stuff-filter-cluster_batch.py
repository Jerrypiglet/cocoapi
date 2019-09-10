from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
pylab.rcParams['figure.figsize'] = (8.0, 10.0)

import pycocotools.mask as maskUtils
from matplotlib.patches import Rectangle

from utils import *

dataDir='..'
dataType='train2017'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)
# initialize COCO api for instance annotations
coco_inst=COCO(annFile)

# initialize COCO api for person keypoints annotations
# Format: http://cocodataset.org/#format-data
annFile = '{}/annotations/person_keypoints_{}.json'.format(dataDir,dataType)
coco_kps=COCO(annFile)

# initialize COCO api for caption annotations
annFile = '{}/annotations/captions_{}.json'.format(dataDir,dataType)
coco_caps=COCO(annFile)

# initialize COCO STUFF api for caption annotations
annFile = '%s/annotations/stuff_%s.json' % (dataDir, dataType)
coco_stuff = COCO(annFile)

# display COCO **instance** categories and supercategories
cats = coco_inst.loadCats(coco_inst.getCatIds())
nms=[cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))
nms = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n{}'.format(' '.join(nms)))

# Display COCO **stuff** categories and supercategories
categories = coco_stuff.loadCats(coco_stuff.getCatIds())
categoryNames = [cat['name'] for cat in categories]
print 'COCO Stuff leaf categories: \n', ' '.join(categoryNames)
superCategoryNames = sorted(set([cat['supercategory'] for cat in categories]))
print 'COCO Stuff super categories: \n', ' '.join(superCategoryNames)



if_vis = False
# split_name = 'imgs_with_morethan2_standing_persons'
split_name = 'imgs_with_morethan2_standing_persons_train2017'

import imageio
from tqdm import tqdm

# get all images containing given categories, select one at random
catIds = coco_inst.getCatIds(catNms=['person']);
imgIds = coco_inst.getImgIds(catIds=catIds );
# print(imgIds)
# imgIds = coco.getImgIds(imgIds = [324158])

count_total_obj = 0
size_list = []
valid_img_count = 0
for ImgIdx, imgID in tqdm(enumerate(imgIds)):
    # if valid_img_count%100 == 0:
    #     if_vis = True
    # else:
    #     if_vis = False
    img = coco_inst.loadImgs(imgID)[0]

    # load and display instance annotations
    annIds = coco_inst.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    anns = coco_inst.loadAnns(annIds)
#     coco.showAnns(anns)
    anns_kps = coco_kps.loadAnns(annIds)
    anns_caps = coco_caps.loadAnns(coco_caps.getAnnIds(imgIds=img['id']))
    annIds_allCats = coco_inst.getAnnIds(imgIds=img['id'], iscrowd=None)
    anns_allCats = coco_inst.loadAnns(annIds_allCats)
    annIds_stuff = coco_stuff.getAnnIds(imgIds=img['id'])
    anns_stuff = coco_stuff.loadAnns(annIds_stuff)

    any_clear = False
    num_clear = 0
    num_valid = 0
    bbox_merged_list = []
    if_clear_list = []
    ratio_list = []
    for idx in range(len(anns)):
        ## Get kps
        ann_kps = anns_kps[idx]
        if_clear = check_clear(ann_kps) # check for head up and foot down person
        
        ## Get bbox
        ann = anns[idx]
        mask1 = ann['segmentation']
        # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py#L265
        rle = maskUtils.frPyObjects(mask1, img['height'], img['width'])
        area = maskUtils.area(rle)
        bboxes = maskUtils.toBbox(rle) # [x, y, w, h]
        if len(bboxes.shape)!=2:
#             print('Warning!! len(bboxes.shape)!=2')
            continue     
        bbox_merged = bboxes[0] if len(bboxes)==1 else merge_bboxes(bboxes)
        
        if bbox_merged[2] == 0. or bbox_merged[3] == 0.:
            continue
        ratio = float(bbox_merged[3]) / float(bbox_merged[2])
#         if ratio <= 2.:
#             continue
        
        if if_clear:
            any_clear = True
            num_clear += 1
            if ratio > 2. and ratio < 8.:
                num_valid += 1
        bbox_merged_list.append(bbox_merged)
        if_clear_list.append(if_clear)
        ratio_list.append(ratio)
    
    if any_clear == False:
        continue
    if num_valid < 2:
        continue
    
    ## load and display image
    # I = io.imread('%s/images/%s/%s'%(dataDir,dataType,img['file_name']))
    # use url to load image
    I = io.imread(img['coco_url'])
    
    if if_vis:
    #     plt.figure(figsize=(10, 10))
        plt.figure(figsize=(30, 40))
        plt.subplot(121)
        plt.title('%s'%ImgIdx)
    # #     plt.imshow(I); plt.axis('off')
        plt.imshow(I); 
    #     plt.axis('off')

        coco_kps.showAnns(anns_kps)
        coco_caps.showAnns(anns_caps)
        coco_inst.showAnns(anns_allCats)
    
        plt.subplot(122)
        plt.imshow(I); plt.axis('off')
        
    stuff_id_map = np.zeros((I.shape[0], I.shape[1]), dtype=np.int)
    for ann in anns_stuff:
        color = coco_stuff.showAnns([ann])
        cat = categories[ann['category_id']-92]['name']
        cat_sup = categories[ann['category_id']-92]['supercategory']
        if cat_sup != 'other':
            bbox = ann['bbox']
            plt.text(bbox[0]+bbox[2]/2., bbox[1]+bbox[3]/2., '%s/%s'%(cat_sup, cat), 
                 fontsize=12, color='w', bbox=dict(facecolor=color[0], alpha=0.5, edgecolor='w'))
            
        mask = maskUtils.decode(ann['segmentation'])
        stuff_id_map += mask * (ann['category_id']-92)
    
    if if_vis:
        plt.subplot(121)
        ax = plt.gca()
    H, W = I.shape[0], I.shape[1]
    bbox_valid_list = []
    for bbox_merged, if_clear, ratio in zip(bbox_merged_list, if_clear_list, ratio_list):
        if not if_clear:
            continue
        if_valid = if_clear
        if_ratio = ratio > 2. and ratio < 8.
        if_valid = if_clear and if_ratio
        
        if if_vis:
            rect = Rectangle((bbox_merged[0], bbox_merged[1]), bbox_merged[2], bbox_merged[3], linewidth=2, edgecolor='lime' if if_valid else 'r', facecolor='none')
            # Add the patch to the Axes
            ax.add_patch(rect)
            plt.text(bbox_merged[0], bbox_merged[1]-12, '%.2f'%ratio, 
                     fontsize=12, bbox=dict(facecolor='lime' if if_valid else 'red', alpha=0.5))
        
        lower_left = [np.clip(int(bbox_merged[0]), 0, W-1), np.clip(int(bbox_merged[1]+bbox_merged[3]), 0, H-1)]
        lower_right = [np.clip(int(bbox_merged[0]+bbox_merged[2]), 0, W-1), np.clip(int(bbox_merged[1]+bbox_merged[3]), 0, H-1)]
        lower_left_id = stuff_id_map[lower_left[1], lower_left[0]]
        lower_left_id_subs = [categories[lower_left_id]['supercategory'], categories[lower_left_id]['name']] if lower_left_id!=0 else ['null', 'null']
        lower_right_id = stuff_id_map[lower_right[1], lower_right[0]]
        lower_right_id_subs = [categories[lower_right_id]['supercategory'], categories[lower_right_id]['name']] if lower_right_id!=0 else ['null', 'null']
        if_valid_surface = check_valid_surface(lower_left_id_subs) or check_valid_surface(lower_right_id_subs)
    
        if if_vis:
            plt.text(bbox_merged[0], bbox_merged[1]+bbox_merged[3]+6, '%s/%s'%(lower_left_id_subs[0], lower_left_id_subs[1]),  
                     fontsize=12, bbox=dict(facecolor='lime' if if_valid_surface else 'red', alpha=0.5))
            plt.text(bbox_merged[0], bbox_merged[1]+bbox_merged[3]+18, '%s/%s'%(lower_right_id_subs[0], lower_right_id_subs[1]),  
                     fontsize=12, bbox=dict(facecolor='lime' if if_valid_surface else 'red', alpha=0.5))
            
        if if_valid:
            bbox_valid_list.append(bbox_merged) # [x, y, w, h]
        
    if if_vis:
        plt.savefig('/Users/ruzhu/Documents/Results/%s/%06d_%s.png'%(split_name, ImgIdx, imgID), bbox_inches='tight')
        plt.show()
        clear_output()
        
    bboxes_valid = np.vstack(bbox_valid_list)
    np.save('/Users/ruzhu/Documents/Results/%s/%06d_%s_bboxes_valid'%(split_name, ImgIdx, imgID), bboxes_valid)
    
    valid_img_count = valid_img_count + 1