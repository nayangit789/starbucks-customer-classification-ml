def predict(model, sample):
    result = model.predict([sample])
    return result[0]