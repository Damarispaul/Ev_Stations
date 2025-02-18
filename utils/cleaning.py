import re
from datetime import datetime, timedelta

def clean_time_string(time_str):
    """Clean and standardize time string format."""
    if not isinstance(time_str, str):
        return None
    
    # Remove spaces, dots, and convert to lowercase
    time_str = time_str.strip().lower().replace('.', ':').replace(' ', '')
    
    # Remove any duplicate colons
    while '::' in time_str:
        time_str = time_str.replace('::', ':')
        
    return time_str

def standardize_time(time_str):
    """Convert any time string to standard format HH:MM AM/PM."""
    time_str = clean_time_string(time_str)
    if not time_str:
        return None
    
    # Extract period (AM/PM)
    if 'am' in time_str:
        period = 'AM'
        time_str = time_str.replace('am', '')
    elif 'pm' in time_str:
        period = 'PM'
        time_str = time_str.replace('pm', '')
    else:
        period = 'AM'  # Default to AM if no period specified
    
    # Handle different time formats
    try:
        if ':' in time_str:
            # Split manually instead of using map
            parts = time_str.split(':')
            hour = int(parts[0])
            minute = int(parts[1]) if len(parts) > 1 else 0
        else:
            hour = int(time_str)
            minute = 0
            
        # Validate hours and minutes
        hour = hour % 12 if hour > 12 else hour
        if hour == 0:
            hour = 12
        minute = min(59, minute)
        
        return f"{hour}:{minute:02d} {period}"
    except ValueError:
        return None

def parse_time_range(time_range):
    """Parse a time range and return start and end times."""
    if not isinstance(time_range, str):
        return None, None
    
    # Clean input
    time_range = time_range.strip().lower()
    
    # Split on hyphen and clean
    parts = [p.strip() for p in re.split(r'[-–—]+', time_range)]
    if len(parts) != 2:
        return None, None
    
    start_time, end_time = parts
    
    # If start time doesn't have am/pm, check end time period
    if not any(x in start_time for x in ['am', 'pm']):
        if 'pm' in end_time:
            # If end time is PM, assume start time is AM
            start_time = start_time + 'am'
        else:
            # Otherwise, use the same period as end time
            period = 'am' if 'am' in end_time else 'pm'
            start_time = start_time + period
    
    # Standardize times
    start_time = standardize_time(start_time)
    end_time = standardize_time(end_time)
    
    if not start_time or not end_time:
        return None, None
    
    try:
        start_dt = datetime.strptime(start_time, '%I:%M %p')
        end_dt = datetime.strptime(end_time, '%I:%M %p')
        
        # Handle overnight shifts
        if end_dt < start_dt:
            end_dt += timedelta(days=1)
            
        return start_dt, end_dt
    except ValueError:
        return None, None

def extract_day_and_hours(text):
    """Extract day type and hours from text description."""
    if not isinstance(text, str):
        return {"Day": "business", "Hours": 8}
    
    text = text.lower()
    
    # Time pattern remains the same
    time_pattern = r'(\d{1,2}(?::\d{2})?(?:\s*[ap]m)?(?:\s*[-–—]\s*)\d{1,2}(?::\d{2})?(?:\s*[ap]m))'
    
    # Specific day patterns
    specific_day_pattern = r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday|mon|tue|wed|thu|fri|sat|sun)\b'
    
    # Business-related patterns
    business_pattern = r'''
    \b(m-f|mon-fri|monday-friday|business hours|fleet|employee|
    service|customer|tenant|visitor|guest|student|faculty|staff|
    church|zoo|park|hotel|restaurant|museum|store|dealership|garage|
    lot|market|college|school|university|campus|government|state|city)\b'''
    
    # Check for 24-hour operation
    if re.search(r'24\s*hours|24/7|24-7', text):
        return {"Day": "daily", "Hours": 24}
    
    # First check for specific days
    day_match = re.search(specific_day_pattern, text)
    if day_match:
        specific_day = day_match.group(1)
        # Map abbreviated days to full names
        day_mapping = {
            'mon': 'monday', 'tue': 'tuesday', 'wed': 'wednesday',
            'thu': 'thursday', 'fri': 'friday', 'sat': 'saturday',
            'sun': 'sunday'
        }
        return {"Day": day_mapping.get(specific_day, specific_day), "Hours": 8}
    
    # Then check for business patterns
    business_match = re.search(business_pattern, text)
    if business_match:
        return {"Day": "business", "Hours": 8}
    
    # Extract and parse time range
    time_match = re.search(time_pattern, text)
    if time_match:
        time_range = time_match.group(1)
        start_time, end_time = parse_time_range(time_range)
        
        if start_time and end_time:
            duration = (end_time - start_time).seconds / 3600
            return {"Day": "business", "Hours": duration}  # Default to business if no specific day
    
    # Default case
    return {"Day": "business", "Hours": 8}