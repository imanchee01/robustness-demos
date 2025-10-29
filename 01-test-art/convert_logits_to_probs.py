import json, math

def softmax_row(z):
    # numerically stable softmax
    m = max(z)
    exps = [math.exp(v - m) for v in z]
    s = sum(exps)
    return [e / s for e in exps]

def convert_file(src="audit_out/robuscope_predictions.json",
                 dst="audit_out/robuscope_predictions_probs.json"):
    with open(src, "r") as f:
        data = json.load(f)

    def convert_list(lst):
        for item in lst:
            if "prediction" not in item:
                continue
            pred = item["prediction"]
            # if already looks like probs, skip
            if all(0.0 <= p <= 1.0 for p in pred) and 0.99 <= sum(pred) <= 1.01:
                continue
            item["prediction"] = softmax_row(pred)

    if isinstance(data, list):
        convert_list(data)
    elif isinstance(data, dict):
        for k, v in list(data.items()):
            if k == "meta_information":
                continue
            if isinstance(v, list):
                convert_list(v)

    with open(dst, "w") as f:
        json.dump(data, f)
    print(f"✅ wrote: {dst}")

if __name__ == "__main__":
    convert_file()
