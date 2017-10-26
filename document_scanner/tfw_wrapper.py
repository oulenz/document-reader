import cv2
import numpy as np
import pandas as pd


def label_img(img, model, label_dict, pretrained_client = None):
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if pretrained_client is not None:
        X = np.array([pretrained_client.predict(img)])
    else:
        X = [img]
    yhat = model.predict(X = X)

    return label_dict[np.argmax(yhat)]


def classify_img_series(img_series, model):
    h, w, c = model.X_shape
    img_series = img_series.apply(lambda x: cv2.resize(x, (w, h)))
    if c == 3:
        img_series = img_series.apply(lambda x: cv2.cvtColor(x, cv2.COLOR_GRAY2RGB))

    img_array = np.array(img_series.tolist())
    n, h, w = img_array.shape[:3]
    img_array = np.reshape(img_array, [n, h, w, c])

    yhat = pd.Series(model.predict(img_array).tolist(), index=img_series.index)

    return yhat.apply(lambda x: np.argmax(x))


def label_image_df(crop_df, model_df):
    field_df = pd.DataFrame()

    for model_name, fields_of_model_df in crop_df.groupby(['model_name']):
        model, label_dict = model_df.loc[model_name, ['model', 'label_dict']]
        #TODO: build in predict as label into tfwrapper
        classes = classify_img_series(fields_of_model_df['crop'], model)
        labels = classes.apply(lambda x: label_dict[x])
        new_content = pd.DataFrame(fields_of_model_df['crop'].copy())
        new_content['label'] = labels
        field_df = field_df.append(new_content)

    return field_df