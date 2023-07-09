import json

# fn = "/root/nyu/project/YOLOX/datasets/dataset/tb/COCO/annotations/instances_train2017_v3.json"
# with open('/root/nyu/project/YOLOX/datasets/dataset/tb/COCO/annotations/instances_train2017_v2.json','r') as load_f:
fn = "/root/nyu/project/YOLOX/datasets/dataset/tb/COCO/annotations/instances_val2017_v3.json"

with open('/root/nyu/project/YOLOX/datasets/dataset/tb/COCO/annotations/instances_val2017.json','r') as load_f:
    data = json.load(load_f)
    categories = data['categories']
    modified_categories = [category for category in categories if category['id'] == 47 or category['id'] == 77]
    for category in modified_categories:
        if category['id'] == 47:
            category['id'] = 0
        else:
            category['id'] = 1
# 将修改后的数据更新回原始数据
    data['categories'] = modified_categories
    
    annotations = data['annotations']
    modified_annotations = [annotation for annotation in annotations if annotation['category_id'] == 47 or  annotation['category_id'] == 77]
    for annotation in modified_annotations:
        if annotation['category_id'] == 47:
            annotation['category_id'] = 0
        else:
            annotation['category_id'] = 1
    data['annotations'] = modified_annotations

# 将修改后的数据转换回JSON字符串
    modified_json = json.dumps(data)

    with open(fn, "w") as file:
        file.write(modified_json)


    



