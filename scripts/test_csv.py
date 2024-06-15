import pandas as pd
def gen_csv(num_rows,output_path):
    # Define the number of rows needed based on the provided screenshot (assuming 20 rows here as an example)

    # Create a DataFrame with the desired path values
    df = pd.DataFrame({
        'path': [f'test/{str(i).zfill(5)}.flac' for i in range(num_rows)]
    })

    # Save the DataFrame to a CSV file in the specified directory
    df.to_csv(output_path, index=False)

if __name__ == '__main__':
    num_rows=[3803,100,3926,20,140,1898]
    gen_csv(num_rows[5],"/home/fanshitong/run/acoustic-traffic-simulation-counting-main/data_root/real_root/loc6/test.csv")
