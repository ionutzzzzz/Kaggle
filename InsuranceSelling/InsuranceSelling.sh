clear
echo "  Running the script . . .                            "
echo "======================================================"
echo " _______  ____   ____                     _____       "      
echo "    |    |    | |    | | |\\   | | |\\   | |          "
echo "    |    |____| |____| | | \\  | | | \\  | |  ___     "
echo "    |    |\\     |    | | |  \\ | | |  \\ | |     |   "
echo "    |    | \\    |    | | |   \\| | |   \\| |_____|   "
echo "                                                      "
echo "======================================================"
python3 train.py
clear
echo "  Model is finished . . .                             "
echo "======================================================"
echo "           _____   ____     ______                    "      
echo " |\\    /| |     | |    \   |        |                "
echo " | \\  / | |     | |     \  |        |                "
echo " |  \\/  | |     | |      | |----    |                "
echo " |      | |     | |     /  |        |                 "
echo " |      | |_____| |____/   |______  |_____            "
echo "                                                      "
echo "======================================================"
cat model/model.log
echo ""
echo "Making the submission prediction . . ."
python3 test.py
echo ""
echo "Shutting down server . . ."
echo ""
# sudo shutdown -h now