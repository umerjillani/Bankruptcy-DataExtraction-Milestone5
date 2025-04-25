# Bankruptcy Case Data Extractor

This script processes bankruptcy case data from CSV files, extracting key information about cases, legal representatives, and court details. It uses a combination of regex patterns and OpenAI's GPT model to accurately extract and clean up legal information.

## Features

- Extracts case details from bankruptcy court dockets
- Identifies and processes legal representatives (DIP1, DIP2, CRO, etc.)
- Handles duplicate detection to prevent reprocessing
- Uses OpenAI for intelligent text extraction and cleanup
- Real-time processing with immediate output file updates
- Comprehensive logging for debugging and tracking

## Prerequisites

- Python 3.x
- OpenAI API key
- Required Python packages (see requirements.txt)

## Installation

1. Clone the repository or download the script files
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the project root with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Project Structure

```
project_root/
├── script.py              # Main processing script
├── requirements.txt       # Python dependencies
├── .env                  # Environment variables (create this)
├── Input Folder/         # Input folder for CSV files
└── NewTrackerSheet0.csv  # Output file (created automatically)
```

## Usage

1. Place your CSV files in the `Input Folder` directory
2. Run the script:
   ```bash
   python script.py
   ```

The script will:
- Process all CSV files in the `Input Folder`
- Extract case details and legal information
- Append new entries to `NewTrackerSheet0.csv`
- Skip already processed files
- Provide detailed logging of the process

## Changing Folder Paths

To modify the input or output folder paths:

1. Open `script.py` in a text editor
2. Locate the main execution block at the bottom of the file:
   ```python
   if __name__ == "__main__":
       main_folder = "Input Folder"  # Change this to your desired input folder path
       output_file = "NewTrackerSheet0.csv"  # Change this to your desired output file path
       process_folder(main_folder, output_file)
   ```
3. Modify the `main_folder` variable to point to your desired input folder
4. Modify the `output_file` variable to specify your desired output file path
5. Save the changes and run the script

Note: When changing paths:
- Use forward slashes (/) or double backslashes (\\) in Windows paths
- Ensure the specified folders exist before running the script
- Use absolute paths if the folders are not in the same directory as the script

## Data Extraction Details

The script extracts the following information from each case:

- Case Number
- Court Location
- Judge Name
- Claims Agent
- Chief Restructuring Officer (CRO)
- Financial Advisor/Investment Banker
- DIP1 Counsel
- DIP2 Counsel
- Committee Counsel (Primary and Secondary)
- Committee Financial Advisors
- Confirmation Hearing Date

## Processing Logic

1. **File Processing**:
   - Reads CSV files from the `Input Folder`
   - Extracts company name from filename
   - Processes docket entries for case details

2. **Duplicate Detection**:
   - Normalizes company names and case numbers
   - Maintains a set of processed entries
   - Skips files that have already been processed

3. **Data Extraction**:
   - Uses regex patterns for initial extraction
   - Employs OpenAI for complex text analysis
   - Cleans and normalizes extracted data

4. **Output Management**:
   - Appends new entries to the output file
   - Maintains data consistency
   - Provides real-time updates

## Error Handling

The script includes comprehensive error handling for:
- File reading/writing errors
- OpenAI API errors
- Data extraction failures
- Invalid file formats

## Logging

The script provides detailed logging of:
- Processing status
- File operations
- Data extraction results
- Error messages
- Duplicate detection

## Notes

- The script uses OpenAI's GPT-3.5-turbo model for text analysis
- Processing time may vary based on file size and complexity
- Ensure sufficient OpenAI API credits for processing
- The script maintains data integrity by checking for duplicates

## Support

For issues or questions, please check the error logs and ensure:
1. All prerequisites are properly installed
2. The OpenAI API key is valid
3. Input files are in the correct format
4. The output file is not locked by another process
5. The input and output paths are correctly specified 