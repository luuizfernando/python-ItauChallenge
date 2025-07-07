from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import numpy as np
import joblib
import os
from dateutil import parser

model_path = os.path.join(os.getcwd(), 'model', 'fraud_model.pkl')
model = joblib.load(model_path)

@csrf_exempt
def analyze_fraud(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            valor = float(data["valor"])
            dataHora = parser.parse(data["dataHora"])

            hour = dataHora.hour
            isLateNight = 1 if 0 <= hour <= 5 else 0
            isHighValue = 1 if valor > 5000 else 0

            entry = np.array([[valor, hour, isLateNight, isHighValue]])

            result = model.predict(entry)[0]
            response = "fraud" if result == 1 else "legal"

            return JsonResponse({"Result": response})

        except Exception as e:
            return JsonResponse({"Error": str(e)}, status=400)

    return JsonResponse({"Error": "Method not allowed"}, status=405)