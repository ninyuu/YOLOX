import fiftyone as fo
from fiftyone import ViewField as F

coco_val_anno = '/root/nyu/project/YOLOX/datasets/dataset/coco/coco2017/annotations/instances_val2017.json'
coco_val_img = '/root/nyu/project/YOLOX/datasets/dataset/coco/coco2017/val2017'


dataset_coco_val = fo.Dataset.from_dir(
    dataset_type=fo.types.COCODetectionDataset,
    data_path=coco_val_img,
    labels_path=coco_val_anno,
    include_id=True,
    classes=['car', 'truck', 'bus', 'person'] # 只选取包含这几个类的图片
)

# filtered_coco_val = dataset_coco_val.set_field(
#     "detections.detections.label",
#     (F("label").is_in(['car', 'truck', 'bus'])).if_else("vehicle", F("label"))
# )


bdd_val_anno = '/root/nyu/project/YOLOX/datasets/dataset/bdd100k/bdd100k/labels/100k/labels_coco/bdd100k_labels_images_det_coco_val.json'
bdd_val_img = '/root/nyu/project/YOLOX/datasets/dataset/bdd100k/bdd100k/images/100k/val'
dataset_bdd_val = fo.Dataset.from_dir(
    dataset_type=fo.types.COCODetectionDataset,
    data_path=bdd_val_img,
    labels_path=bdd_val_anno,
    include_id=True,
    # classes=['car', 'truck', 'bus','bike','motor', 'person','rider'] # 只选取包含这几个类的图片,不代表过滤掉了图片中的交通灯类别
    classes=['car', 'truck', 'bus','bike','motor', 'person'] # 只选取包含这几个类的图片,不代表过滤掉了图片中的交通灯类别

)

# filtered_view_bdd_val = dataset_bdd_val.set_field(
#     "detections.detections.label",
#     (F("label").is_in(['car', 'truck', 'bus','bike','motor'])).if_else("vehicle", F("label"))
# )

# filtered_view_bdd_val = dataset_bdd_val.set_field(
#     "detections.detections.label",
#     (F("label").is_in(['person','rider'])).if_else("person", F("label"))
# )
 
# filtered_view1.merge_samples(filtered_view2)

#先合并数据集，再合并类别
dataset_coco_val.merge_samples(dataset_bdd_val)

filtered_view1 = dataset_coco_val.set_field(
    "detections.detections.label",
    (F("label").is_in(['car', 'truck', 'bus','bike','motor'])).if_else("vehicle", F("label"))
)

#test 不要rider
# filtered_view1 = dataset_coco_val.set_field(
#     "detections.detections.label",
#     (F("label").is_in(['person','rider'])).if_else("person", F("label"))
# )


# dataset1.add_samples(filtered_view1)
# session = fo.launch_app(filtered_view1,port=16698)
# session.wait()


export_dir = "/root/nyu/project/YOLOX/datasets/dataset/tb/COCO/val2017"
filtered_view1.export(
    export_dir=export_dir,
    dataset_type=fo.types.COCODetectionDataset,
)



coco_train_anno = '/root/nyu/project/YOLOX/datasets/dataset/coco/coco2017/annotations/instances_train2017.json'
coco_train_img = '/root/nyu/project/YOLOX/datasets/dataset/coco/coco2017/train2017'


dataset_coco_train = fo.Dataset.from_dir(
    dataset_type=fo.types.COCODetectionDataset,
    data_path=coco_train_img,
    labels_path=coco_train_anno,
    include_id=True,
    classes=['car', 'truck', 'bus', 'person'] # 只选取包含这几个类的图片
)

# filtered_coco_val = dataset_coco_val.set_field(
#     "detections.detections.label",
#     (F("label").is_in(['car', 'truck', 'bus'])).if_else("vehicle", F("label"))
# )


bdd_train_anno = '/root/nyu/project/YOLOX/datasets/dataset/bdd100k/bdd100k/labels/100k/labels_coco/bdd100k_labels_images_det_coco_train.json'
bdd_train_img = '/root/nyu/project/YOLOX/datasets/dataset/bdd100k/bdd100k/images/100k/train'
dataset_bdd_train = fo.Dataset.from_dir(
    dataset_type=fo.types.COCODetectionDataset,
    data_path=bdd_train_img,
    labels_path=bdd_train_anno,
    include_id=True,
    classes=['car', 'truck', 'bus','bike','motor', 'person','rider'] # 只选取包含这几个类的图片,不代表过滤掉了图片中的交通灯类别
)

# filtered_view_bdd_val = dataset_bdd_val.set_field(
#     "detections.detections.label",
#     (F("label").is_in(['car', 'truck', 'bus','bike','motor'])).if_else("vehicle", F("label"))
# )

# filtered_view_bdd_val = dataset_bdd_val.set_field(
#     "detections.detections.label",
#     (F("label").is_in(['person','rider'])).if_else("person", F("label"))
# )
 
# filtered_view1.merge_samples(filtered_view2)

#先合并数据集，再合并类别
dataset_coco_train.merge_samples(dataset_bdd_train)

filtered_view2 = dataset_coco_train.set_field(
    "detections.detections.label",
    (F("label").is_in(['car', 'truck', 'bus','bike','motor'])).if_else("vehicle", F("label"))
)
# filtered_view2 = dataset_coco_train.set_field(
#     "detections.detections.label",
#     (F("label").is_in(['person','rider'])).if_else("person", F("label"))
# )
# dataset1.add_samples(filtered_view1)
# session = fo.launch_app(filtered_view1,port=16698)
# session.wait()


export_dir = "/root/nyu/project/YOLOX/datasets/dataset/tb/COCO/train2017"
filtered_view2.export(
    export_dir=export_dir,
    dataset_type=fo.types.COCODetectionDataset,
)