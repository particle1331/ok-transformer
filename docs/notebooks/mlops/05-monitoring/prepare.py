from tqdm import tqdm
import requests

files = [
    ("fhv_tripdata_2021-01.parquet", "./evidently_service/datasets"),
    ("fhv_tripdata_2021-12.parquet", ".")
]

print(f"Download files:")
for file, path in files:
    url = f"https://raw.githubusercontent.com/alexeygrigorev/datasets/master/nyc-tlc/fhv/{file}"
    resp = requests.get(url, stream=True)
    save_path = f"{path}/{file}"
    with open(save_path, "wb") as handle:
        for data in tqdm(
            resp.iter_content(),
            desc=f"{file}",
            postfix=f"save to {save_path}",
            total=int(resp.headers["Content-Length"])
        ):
            handle.write(data)
