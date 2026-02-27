import pandas as pd

def load_data(red_path, white_path):
    red = pd.read_csv(red_path, sep=';')       
    white = pd.read_csv(white_path, sep=';') 
    red['wine_type'] = 'red'          
    white['wine_type'] = 'white'      

    df = pd.concat([red, white], ignore_index=True)  
    return df

if __name__ == "__main__":
    df = load_data("/Users/camicecegussoni/Desktop/Wine_project/dataset/winequality-red.csv",
                   "/Users/camicecegussoni/Desktop/Wine_project/dataset/winequality-white.csv")
    print(df.head())
