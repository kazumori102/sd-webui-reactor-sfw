from transformers import pipeline
from PIL import Image
from functools import lru_cache

SCORE = 0.965 # 0.965 and less - is safety content


@lru_cache(maxsize=2)
def _classifier(model_path: str):
    return pipeline("image-classification", model=model_path, device=-1)

def nsfw_image(img_path: str, model_path: str):
    # Set the pipeline's device through its public API; keep this small
    # classifier off Forge's diffusion-model GPU allocation.
    predict = _classifier(model_path)
    with Image.open(img_path) as img:
        result = predict(img, top_k=None)
    # Transformers sorts by confidence. The top result can be the SAFE class.
    for prediction in result:
        if prediction["label"].lower() == "nsfw":
            score = prediction["score"]
            print(f"NSFW Score = {score}")
            return score > SCORE
    raise ValueError("SFW detector did not return an NSFW class; processing stopped")
