import fiftyone as fo
from fiftyone import ViewField as F

#以下是以COCO2017 val验证集，将几个车辆类合并为一个vehicle类的例子。（fiftyone不会修改原始数据集，可以放心操作）
# 本机上的COCO数据集路径
# 这里以val验证集为例
coco_anno = '/root/nyu/project/YOLOX/datasets/dataset/DETRAC/annotations/instances_val2017.json'
coco_img = '/root/nyu/project/YOLOX/datasets/dataset/DETRAC/val2017'

dataset = fo.Dataset.from_dir(
    dataset_type=fo.types.COCODetectionDataset,
    data_path=coco_img,
    labels_path=coco_anno,
    include_id=True,
    classes=['car', 'van', 'bus', 'others'] # 只选取包含这几个类的图片
)

# 把'car', 'truck', 'bus'三个类合并为 'vehicle'
filtered_view = dataset.set_field(
    "detections.detections.label",
    (F("label").is_in(['car', 'van', 'bus','others'])).if_else("vehicle", F("label"))
)

# 在网页上预览数据集
session = fo.launch_app(dataset,port=16698)
session.wait()

#### 导出数据集 ####
# export_dir = "/root/nyu/project/YOLOX/datasets/dataset/tb/detrack/val2017"
# filtered_view.export(
#     export_dir=export_dir,
#     dataset_type=fo.types.COCODetectionDataset,
# )
###################


