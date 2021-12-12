import tflite_runtime.interpreter as tflite
from keras_image_helper import create_preprocessor


preprocessor = create_preprocessor('xception', target_size=(299, 299))


interpreter = tflite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']


classes =  ['banana_legs',
            'beefsteak',
            'blue_berries',
            'cherokee_purple',
            'german_orange_strawberry',
            'green_zebra',
            'japanese_black_trifele',
            'kumato',
            'oxheart',
            'roma',
            'san_marzano',
            'sun_gold',
            'supersweet_100',
            'tigerella',
            'yellow_pear']


def predict(url):
    X = preprocessor.from_url(url)

    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)

    float_predictions = preds[0].tolist()

    return dict(zip(classes, float_predictions))


def lambda_handler(event, context):
    url = event['url']
    result = predict(url)
    return result


