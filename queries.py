import requests
from bs4 import BeautifulSoup
import pandas as pd


def get_historical_trades(session, nasdaq_id: str, from_date: str = "2022-01-01", to_date: str="2023-01-01", save_to_file=False):
    print(f"Fetching id: {nasdaq_id}, from {from_date}, to {to_date}")

    headers = {
        'authority': 'www.nasdaqomxnordic.com',
        'accept': '*/*',
        'accept-language': 'da-DK,da;q=0.8',
        'cache-control': 'no-cache',
        'content-type': 'application/x-www-form-urlencoded; charset=UTF-8',
        'origin': 'https://www.nasdaqomxnordic.com',
        'pragma': 'no-cache',
        'referer': 'https://www.nasdaqomxnordic.com/aktier/microsite?Instrument=CSE3200&name=A.P.%20M%C3%B8ller%20-%20M%C3%A6rsk%20A&ISIN=DK0010244425',
        'sec-ch-ua': '"Chromium";v="116", "Not)A;Brand";v="24", "Brave";v="116"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"macOS"',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'same-origin',
        'sec-gpc': '1',
        'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36',
        'x-requested-with': 'XMLHttpRequest',
    }

    data = {
        'xmlquery': f'<post>\n<param name="SubSystem" value="Prices"/>\n<param name="Action" value="GetTrades"/>\n<param name="Exchange" value="NMF"/>\n<param name="t__a" value="30,38,31,5,32,39,33,26,40,35,36,28,99,1,2,7,8,18"/>\n<param name="FromDate" value="{from_date}"/>\n<param name="ToDate" value="{to_date}"/>\n<param name="Instrument" value="{nasdaq_id}"/>\n<param name="ext_contenttype" value="application/ms-excel"/>\n<param name="ext_contenttypefilename" value="share_export.csv"/>\n<param name="ext_xslt" value="/nordicV3/trades_csv.xsl"/>\n<param name="ext_xslt_lang" value="da"/>\n<param name="showall" value="1"/>\n<param name="app" value="/aktier/microsite"/>\n</post>',
    }

    response = session.post('https://www.nasdaqomxnordic.com/webproxy/DataFeedProxy.aspx',
                             headers=headers, data=data)
    if "No data available right now" in response.text:
        print(f"No trades for id: {nasdaq_id}")
        return None

    if not response.ok:
        print(f"Bad response code!: {response.text}")
        return None

    assert response.text[:7] == "sep=;\r\n"

    trades = response.text[7:]

    trades = clean_historical_trades(trades=trades)

    # Save to file
    if save_to_file:
        file_name = nasdaq_id
        with open(f"{helpers.root_path()}/data/historical_trades/{file_name}.csv", mode="w") as f:
            f.write(trades)
            print(f"{nasdaq_id} dumped to file!")
    return trades


def clean_historical_trades(trades=None):
    df = pd.DataFrame([x.split(';') for x in trades.split('\r\n')])
    df = df.rename(columns=df.iloc[0]).drop(df.index[0])    # Convert first row to column names
    df = df.iloc[:-2, :]    # Drop bottom empty row
    cols = ['Price', 'Volume', 'Execution Time UTC']
    try:
        df = df[cols]
    except KeyError:
        print(f"Couldnt find columns in {df}")
        return None
    df['Execution Time UTC'] = pd.to_datetime(df['Execution Time UTC'])
    df['Execution Time UTC'].interpolate(inplace=True, method="ffill")
    df['Trade_date'] = df['Execution Time UTC'].dt.date

    df['Price'] = df['Price'].astype("float")
    df['Volume'] = df['Volume'].astype("int")

    df = df.rename(columns={
    'Trade type': 'Trade_type',
    'Execution Time UTC': 'Trade_time'})
    df = df.sort_values('Trade_time')
    df = df.reset_index(drop=True)
    return df


def remove_outliers(df):
    # Compute percentage changes
    df['pct_change'] = df['Price'].pct_change().abs()

    # Define a threshold for large percentage changes
    avg_std = df["Price"].std()*100/df["Price"].mean()
    magic_constant = 0.01
    threshold = magic_constant*(avg_std**1.5)

    # Identify rows where a large percentage change is followed by a similarly large percentage change in the opposite direction
    mask = (
        (df['pct_change'].shift(-1) > threshold) &
        (df['pct_change'] > threshold))

    # Drop the identified rows and the 'pct_change' column
    df = df[~mask].drop(columns=['pct_change'])
    return df

def get_all_nasdaq_ids():
    headers = {
        'authority': 'www.nasdaqomxnordic.com',
        'accept': 'text/html, */*; q=0.01',
        'accept-language': 'da-DK,da;q=0.8',
        'cache-control': 'no-cache',
        'content-type': 'application/x-www-form-urlencoded; charset=UTF-8',
        'origin': 'https://www.nasdaqomxnordic.com',
        'pragma': 'no-cache',
        'referer': 'https://www.nasdaqomxnordic.com/aktier',
        'sec-ch-ua': '"Chromium";v="116", "Not)A;Brand";v="24", "Brave";v="116"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"macOS"',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'same-origin',
        'sec-gpc': '1',
        'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36',
        'x-requested-with': 'XMLHttpRequest',
    }

    data = "xmlquery=%3Cpost%3E%0A%3Cparam+name%3D%22Exchange%22+value%3D%22NMF%22%2F%3E%0A%3Cparam+name%3D%22SubSystem%22+value%3D%22Prices%22%2F%3E%0A%3Cparam+name%3D%22Action%22+value%3D%22GetMarket%22%2F%3E%0A%3Cparam+name%3D%22inst__a%22+value%3D%220%2C1%2C2%2C5%2C21%2C23%22%2F%3E%0A%3Cparam+name%3D%22ext_xslt%22+value%3D%22%2FnordicV3%2Finst_table_shares_ge.xsl%22%2F%3E%0A%3Cparam+name%3D%22Market%22+value%3D%22L%3AINET%3AH7053910%2CL%3AINET%3AH7053920%2CL%3AINET%3AH7053930%22%2F%3E%0A%3Cparam+name%3D%22inst__an%22+value%3D%22id%2Csectrid%2Cnm%2Cfnm%2Ccr%2Clsp%2Ctp%2Cch%2Cchp%2Cbp%2Cap%2Ctv%2Cto%2Chlp%2Clists%2Cnote%2Cpmkt%2Cmktc%2Cisin%2Cst%2Cslc%2Cstc%22%2F%3E%0A%3Cparam+name%3D%22inst__e%22+value%3D%2213%22%2F%3E%0A%3Cparam+name%3D%22issuer__e%22+value%3D%224%22%2F%3E%0A%3Cparam+name%3D%22issuer__a%22+value%3D%222%22%2F%3E%0A%3Cparam+name%3D%22Lang%22+value%3D%22da%22%2F%3E%0A%3Cparam+name%3D%22XPath%22+value%3D%22%2F%2Finst%5B%40tp%3D'S'+or+%40tp%3D'ER'+and+%40st!%3D'Subscription+right'%5D%22%2F%3E%0A%3Cparam+name%3D%22ext_xslt_sortattribute%22+value%3D%22fnm%22%2F%3E%0A%3Cparam+name%3D%22ext_xslt_lang%22+value%3D%22da%22%2F%3E%0A%3Cparam+name%3D%22ext_xslt_tableId%22+value%3D%22searchSharesListTable%22%2F%3E%0A%3Cparam+name%3D%22ext_xslt_hiddenattrs%22+value%3D%22%2Clists%2Chlp%2Ctp%2Cisin%2Cnote%2Cmktc%2Cpmkt%2Cst%2Cslc%2Cstc%2C%22%2F%3E%0A%3Cparam+name%3D%22ext_xslt_tableClass%22+value%3D%22tablesorter%22%2F%3E%0A%3Cparam+name%3D%22ext_xslt_options%22+value%3D%22%2Cnoflag%2Csectoridicon%2Ctruncate%2C%22%2F%3E%0A%3Cparam+name%3D%22DefaultDecimals%22+value%3D%22false%22%2F%3E%0A%3Cparam+name%3D%22app%22+value%3D%22%2Faktier%22%2F%3E%0A%3C%2Fpost%3E"

    response = requests.post(
        'https://www.nasdaqomxnordic.com/webproxy/DataFeedProxy.aspx',
        headers=headers,
        data=data,
    )

    # Parse the HTML with BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')

    # Iterate over <tr> elements
    ids = [[row['id'][22:], row['title']] for row in soup.find_all('tr', id=True) if row['id'].startswith('searchSharesListTable')]

    return ids