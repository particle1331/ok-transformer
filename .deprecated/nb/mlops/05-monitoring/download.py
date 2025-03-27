from tqdm import tqdm
import requests


files = [
    ("fhv_tripdata_2021-01.parquet", "./evidently_service/datasets"),
    ("fhv_tripdata_2021-12.parquet", ".")
]

host = f"https://raw.githubusercontent.com/alexeygrigorev/datasets/master/nyc-tlc/fhv"
    
print(f"Downloading files:")
for file, path in files:
    response = requests.get(url=f"{host}/{file}", stream=True)
    save_path = f"{path}/{file}"
    
    with open(save_path, "wb") as handle:
        for data in tqdm(
            response.iter_content(),
            desc=f"  {file}",
            postfix=f"{save_path}",
            total=int(response.headers["Content-Length"])
        ):
            handle.write(data)
