import pkg_resources
import pandas as pd


def getPackagesDataframe(list):
    df=pd.DataFrame(columns=["name","version"])

    if(len(list)!=0):
        for element in list:
            newPackage=django_package=pd.DataFrame({"name":[element],"version":[(pkg_resources.get_distribution(element).version)]})
            df = df.append(newPackage, ignore_index=True)
        return df

    print("You are not giving any package to retrieve")

def getPackagesList():
    
    packagesList=["pandas",
                  "django",
                  "tensorflow",
                  "goodtables",
                  "tableschema",
                  "numpy",
                  "pandas_schema",
                  "joblib",
                  "keras",
                  ]
                  
    return packagesList
