import torch
from torchvision.transforms import transforms
from torchvision.datasets import COCO
from yolact import Yolact
from yolact.utils import Timer

# 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 데이터셋 로드
data_transform = transforms.Compose([
    transforms.Resize((550, 550)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_dataset = COCO(root='path_to_validation_dataset', annFile='path_to_annotations', transform=data_transform)

# 모델 불러오기
model = Yolact()
model.load_weights('path_to_pretrained_weights')
model.to(device)
model.eval()

# 정확도 평가
correct = 0
total = 0
timer = Timer()

with torch.no_grad():
    for images, targets in val_dataset:
        images = images.to(device)
        predictions = model(images)

        # 예측 결과 평가
        # 이 부분은 모델마다 예측 결과를 어떻게 평가할지에 따라 다를 수 있습니다.
        # 여기서는 간단히 첫 번째 클래스의 예측이 정확한지를 확인하는 것으로 예시를 들었습니다.
        for i, pred in enumerate(predictions):
            if pred['class_ids'][0] == targets['category_id']:
                correct += 1
            total += 1

        # 경과 시간 출력
        print('Elapsed time: {}'.format(timer.elapsed()))

# 정확도 출력
accuracy = correct / total
print('Accuracy: {:.2f}%'.format(accuracy * 100))
