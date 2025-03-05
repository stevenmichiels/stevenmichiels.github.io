import os
from XSPYForecast import run_forecast
from datetime import datetime

def update_website_forecast():
    # Generate new forecast
    run_forecast(instrument='SPX', start_year=1962, subfolder='stevenmichiels.github.io')
    
    print("✅ Forecast updated successfully")

if __name__ == "__main__":
    update_website_forecast()
