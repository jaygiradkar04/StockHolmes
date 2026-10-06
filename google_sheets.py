import os
import re
from datetime import datetime
from google.oauth2 import service_account
from googleapiclient.discovery import build
import logging

logger = logging.getLogger(__name__)

SCOPES = ['https://www.googleapis.com/auth/spreadsheets']

# Fuzzy mapping dictionary to associate yfinance tickers with user's Google Sheet company names
STOCK_SYNONYMS = {
    'trent.ns': ['trent'],
    'sterling.ns': ['sterling', 'sterling tools', 'sterling generators'],
    'swsolar.ns': ['sterling', 'swsolar', 'sterling and wilson', 'sterling & wilson'],
    'sterlingtools.ns': ['sterling', 'sterling tools', 'sterlingtools'],
    'sunpharma.ns': ['sun pharma. inds.', 'sun pharma', 'sun pharmaceutical', 'sunpharma'],
    'sbin.ns': ['state bank of india', 'state bank', 'sbin', 'sbi'],
    'm&m.ns': ['mahindra & mahindra', 'm&m', 'mahindra'],
    'hdfcbank.ns': ['hdfc bank', 'hdfc'],
    'cochinship.ns': ['cochin shipyard', 'cochin ship', 'cochinship'],
    'cdsl.ns': ['central dep. service', 'cdsl', 'central depository services', 'central depository'],
    'bsoft.ns': ['birlasoft', 'bsoft'],
    'bse.ns': ['bse', 'bse limited'],
    'biocon.ns': ['biocon'],
    'unienter.ns': ['uniphos enterprises', 'uniphos', 'unienter'],
    'salora.ns': ['salora international', 'salora'],
    'saloraintl.bo': ['salora international', 'salora'],
    'goodluck.ns': ['goodluck india', 'goodluck'],
    'ncc.ns': ['ncc'],
    'tatapower.ns': ['tata power', 'tatapower', 'tata'],
    'nmdc.ns': ['nmdc'],
    'stovekraft.ns': ['stove kraft', 'stovekraft'],
    'itc.ns': ['itc ltd', 'itc'],
    'nestleind.ns': ['nestle india', 'nestle', 'nestlé india', 'nestl india'],
    'hindunilvr.ns': ['hindustan unilever (hul)', 'hindunilvr', 'hul', 'hindustan unilever'],
    'tcs.ns': ['tcs', 'tata consultancy services', 'tata consultancy']
}

def get_sheets_service(creds_path='service_account.json'):
    """
    Authenticate with Google Sheets API using service account credentials.
    """
    if not os.path.exists(creds_path):
        return None
    try:
        creds = service_account.Credentials.from_service_account_file(
            creds_path, scopes=SCOPES)
        service = build('sheets', 'v4', credentials=creds)
        return service
    except Exception as e:
        logger.error(f"Failed to authenticate with Google Sheets: {e}")
        return None

def col_idx_to_letter(idx):
    """
    Convert 0-based column index to Excel column letter (e.g., 0 -> A, 1 -> B, 27 -> AB)
    """
    temp = ""
    while idx >= 0:
        temp = chr(idx % 26 + 65) + temp
        idx = (idx // 26) - 1
    return temp

def normalize_name(name):
    if not name:
        return ""
    return re.sub(r'[^a-z0-9]', '', str(name).lower())

def is_matching_company(symbol, company_name):
    """
    Fuzzy match stock symbol to sheet company name.
    """
    sym_lower = symbol.lower()
    comp_normalized = normalize_name(company_name)
    
    # Check synonym list
    for key, synonyms in STOCK_SYNONYMS.items():
        if sym_lower == key or sym_lower.split('.')[0] == key.split('.')[0]:
            for syn in synonyms:
                if normalize_name(syn) in comp_normalized or comp_normalized in normalize_name(syn):
                    return True
                    
    # Default fallback: check if symbol prefix is in company name or vice versa
    sym_prefix = sym_lower.split('.')[0]
    sym_norm = normalize_name(sym_prefix)
    if sym_norm and (sym_norm in comp_normalized or comp_normalized in sym_norm):
        return True
        
    return False

def extract_dates_from_headers(row1):
    """
    Parse date values in row 1 headers, e.g. "Next Week (8/24/2026)" -> "8/24/2026"
    Returns a dict mapping column index -> date string
    """
    dates_map = {}
    for col_idx, cell in enumerate(row1):
        if not cell:
            continue
        cell_str = str(cell).lower()
        if "today" in cell_str or "price" in cell_str:
            continue
        # Extract date like 8/24/2026 or 08/24/2026 or 9/7/2026
        match = re.search(r'(\d{1,2}/\d{1,2}/\d{4})', str(cell))
        if match:
            dates_map[col_idx] = match.group(1)
    return dates_map

def get_earliest_date_in_headers(row1):
    """Find the earliest date present in row 1 headers."""
    dates = []
    for cell in row1:
        if cell:
            match = re.search(r'(\d{1,2}/\d{1,2}/\d{4})', str(cell))
            if match:
                try:
                    dates.append(datetime.strptime(match.group(1), "%m/%d/%Y"))
                except:
                    pass
    return min(dates) if dates else None

def get_first_sheet_title(service, spreadsheet_id):
    """Retrieve the title of the first sheet in the spreadsheet."""
    try:
        spreadsheet = service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
        sheets = spreadsheet.get('sheets', [])
        if sheets:
            return sheets[0].get('properties', {}).get('title', 'Sheet1')
    except Exception as e:
        logger.error(f"Failed to get sheet title: {e}")
    return 'Sheet1'

def sync_stock_data(spreadsheet_id, symbol, today_price, predictions, actuals, sheet_range=None):
    """
    Update row in Google Sheet for stock.
    predictions: { '8/24/2026': 2979.92, ... }
    actuals: { '8/24/2026': 2985.00, ... }
    """
    service = get_sheets_service()
    if not service:
        return False, "service_account_missing"
        
    try:
        sheet_title = get_first_sheet_title(service, spreadsheet_id)
        if not sheet_range:
            sheet_range = f"{sheet_title}!A1:Z100"
        elif sheet_range.startswith("Sheet1!"):
            sheet_range = sheet_range.replace("Sheet1!", f"{sheet_title}!")
            
        # Read the sheet values
        result = service.spreadsheets().values().get(
            spreadsheetId=spreadsheet_id, range=sheet_range).execute()
        rows = result.get('values', [])
        
        if len(rows) < 2:
            return False, "invalid_sheet_structure (need at least 2 header rows)"
            
        row1 = rows[0]
        row2 = rows[1]
        
        # 1. Parse header dates from Row 1
        dates_map = extract_dates_from_headers(row1)
        if not dates_map:
            return False, "no_dates_found_in_headers"
            
        # 2. Find "Today's Price" column
        today_col_idx = -1
        for idx, cell in enumerate(row1):
            if "today" in str(cell).lower():
                today_col_idx = idx
                break
                
        # 3. Find matching company row in sheet
        company_row_idx = -1
        for r_idx in range(2, len(rows)):
            row = rows[r_idx]
            if len(row) > 0 and row[0]:
                if is_matching_company(symbol, row[0]):
                    company_row_idx = r_idx
                    break
                    
        if company_row_idx == -1:
            return False, f"company_not_found_in_sheet: no match found for symbol {symbol}"
            
        # 4. Prepare batch updates
        update_data = []
        
        # Check for stock splits and adjust existing historical cells in this row
        split_ratio = 1.0
        earliest_dt = get_earliest_date_in_headers(row1)
        if earliest_dt:
            try:
                import yfinance as yf
                import pandas as pd
                import numpy as np
                
                ticker = yf.Ticker(symbol)
                splits = ticker.splits
                if not splits.empty:
                    now_tz = splits.index.tz
                    start_dt_tz = pd.Timestamp(earliest_dt).tz_localize(now_tz) if now_tz else pd.Timestamp(earliest_dt)
                    end_dt_tz = pd.Timestamp(datetime.now()).tz_localize(now_tz) if now_tz else pd.Timestamp(datetime.now())
                    
                    recent_splits = splits[(splits.index >= start_dt_tz) & (splits.index <= end_dt_tz)]
                    if not recent_splits.empty:
                        split_ratio = float(np.prod(recent_splits.values))
                        logger.info(f"Detected stock split for {symbol} in date range: ratio {split_ratio}")
            except Exception as e:
                logger.warning(f"Error checking stock splits for {symbol}: {e}")

        if split_ratio != 1.0:
            company_row = rows[company_row_idx]
            for col_idx in range(1, len(company_row)):
                cell_val = company_row[col_idx]
                if cell_val:
                    clean_val = re.sub(r'[^\d\.]', '', str(cell_val))
                    if clean_val:
                        try:
                            old_val = float(clean_val)
                            new_val = old_val / split_ratio
                            cell_range = f"{sheet_title}!{col_idx_to_letter(col_idx)}{company_row_idx + 1}"
                            formatted_val = round(new_val, 2)
                            update_data.append({
                                'range': cell_range,
                                'values': [[formatted_val]]
                            })
                            company_row[col_idx] = str(formatted_val)
                            logger.info(f"Split-adjusting {symbol} cell at col {col_idx}: {old_val} -> {formatted_val}")
                        except ValueError:
                            pass
        
        # Update weekly predictions and actuals
        for header_idx, date_str in dates_map.items():
            # For this date header (e.g. index 2 for "8/24/2026"), check Row 2 columns
            # Next columns corresponding to this header are:
            # - Column header_idx: Prediction
            # - Column header_idx + 1: Actual
            # Wait, verify Row 2 labels
            pred_col = header_idx
            act_col = header_idx + 1
            
            # Write prediction value if available in prediction dict
            if date_str in predictions and predictions[date_str] is not None:
                val = predictions[date_str]
                cell_range = f"{sheet_title}!{col_idx_to_letter(pred_col)}{company_row_idx + 1}"
                update_data.append({
                    'range': cell_range,
                    'values': [[round(val, 2)]]
                })
                
            # Write actual value if available in actuals dict
            if date_str in actuals and actuals[date_str] is not None:
                val = actuals[date_str]
                cell_range = f"{sheet_title}!{col_idx_to_letter(act_col)}{company_row_idx + 1}"
                update_data.append({
                    'range': cell_range,
                    'values': [[round(val, 2)]]
                })
                
        # 5. Execute batch update
        if update_data:
            body = {
                'valueInputOption': 'USER_ENTERED',
                'data': update_data
            }
            service.spreadsheets().values().batchUpdate(
                spreadsheetId=spreadsheet_id, body=body).execute()
            return True, f"Successfully synced data for {symbol}."
        else:
            return False, "no_updates_needed"
            
    except Exception as e:
        logger.error(f"Failed to sync sheet data: {e}")
        return False, str(e)
