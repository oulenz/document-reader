import cv2
import numpy as np
import pandas as pd


def predict_with_model(model, img_series):
    h, w, c = model.X_shape
    img_series = img_series.apply(lambda x: cv2.resize(x, (w, h)))
    if c == 3:
        img_series = img_series.apply(lambda x: cv2.cvtColor(x, cv2.COLOR_GRAY2RGB))

    img_array = np.array(img_series.tolist())
    n, h, w = img_array.shape[:3]
    img_array = np.reshape(img_array, [n, h, w, c])

    yhat = pd.Series(model.predict(img_array).tolist(), index=img_series.index)

    return yhat.apply(lambda x: np.argmax(x))


def classify_images(df_with_images, model_dict):
    content_df = pd.DataFrame()

    for model_name, fields_of_model_df in df_with_images.groupby(['model_name']):
        model, label_dict = model_dict[model_name]
        #TODO: build in predict as label into tfwrapper
        classes = predict_with_model(model, fields_of_model_df['crop'])
        labels = classes.apply(lambda x: label_dict[x])
        new_content = pd.DataFrame(fields_of_model_df['crop'].copy())
        new_content['label'] = labels
        content_df = content_df.append(new_content)

    return content_df