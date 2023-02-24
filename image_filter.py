from PIL import Image
import pandas as pd
import glob
import matplotlib.pyplot as plt

#1250x1250 looks about to be a good choice.
data = {
    "x": list(),
    "y": list()
}

thrown_out = 0
total = 0
threshold = 1250

#Loops through the images and throws out a certain percentage to see how the 
#distribution of image sizes look after throwing out images that exceed the threshold
#bounds. This was used to determine the standard size of our data to be ~1250x1250 for training.
for file in glob.glob("data/Images/**/*.jpg", recursive=True):
    with Image.open(file) as im:
        if im.size[0] <= threshold and im.size[1] <= threshold:
            data["x"].append(im.size[0])
            data["y"].append(im.size[1])
        else:
            thrown_out += 1
    total += 1


df = pd.DataFrame(data)
plt.scatter(df.x, df.y)
print(f"Threw out {thrown_out} images out of {total} images.")
print(f"Throw out percentage: {(thrown_out/total) * 100}%")
plt.show()