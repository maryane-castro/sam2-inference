import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

os.environ["MODEL_CACHE_DIR"] = "./models_cache"
from inference.models.utils import get_model
model_id = "<id_model>"
model = get_model(model_id=model_id, api_key="<api_key>")

image_path = "<path_image>"

# inference
results = model.infer(image_path)[0]

image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

for prediction in results.predictions:  
    points = np.array([(int(point.x), int(point.y)) for point in prediction.points])  
    cv2.polylines(image, [points], isClosed=True, color=(0, 255, 0), thickness=2) 
    cv2.fillPoly(image, [points], color=(0, 255, 0, 50))

    x, y = points[0]
    label = f"{prediction.class_name} ({prediction.confidence:.2f})"
    cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

plt.figure(figsize=(10, 6))
plt.imshow(image)
plt.axis("off")
plt.show()
