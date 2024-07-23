import os
import time


days = 0
hours = 4
minutes = 1
sec = 1
while sec > 0 or minutes > 0 or hours > 0 or days > 0:
    os.system('clear')
    print(f"Time remaining: {days} Days {hours} Hours {minutes} Minutes {sec} Seconds")
    time.sleep(1)
    sec = sec - 1
    if sec == 0:
        sec = 60
        minutes = minutes - 1
        if minutes == 0:
            minutes = 60
            hours = hours - 1
            if hours == 0:
                hours = 24
                days = days - 1
             

os.system('shutdown -h now')
