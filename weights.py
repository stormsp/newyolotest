import torch

# Загружаем веса
weights = torch.load('person.pt', map_location='cpu')
print(weights.keys())
if 'names' in weights:
    print("Классы:", weights['names'])
elif 'model' in weights and hasattr(weights['model'], 'names'):
    print("Классы:", weights['model'].names)
else:
    print("Информация о классах не найдена в файле весов.")
