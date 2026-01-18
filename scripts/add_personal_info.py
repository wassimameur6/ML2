"""
Script to add realistic personal information to the churn dataset.
Adds: First_Name, Last_Name, Email, Phone_Number
"""
import pandas as pd
import random
import string

# Common first names by gender
MALE_NAMES = [
    "James", "John", "Robert", "Michael", "William", "David", "Richard", "Joseph",
    "Thomas", "Christopher", "Charles", "Daniel", "Matthew", "Anthony", "Mark",
    "Donald", "Steven", "Paul", "Andrew", "Joshua", "Kenneth", "Kevin", "Brian",
    "George", "Timothy", "Ronald", "Edward", "Jason", "Jeffrey", "Ryan",
    "Jacob", "Gary", "Nicholas", "Eric", "Jonathan", "Stephen", "Larry", "Justin",
    "Scott", "Brandon", "Benjamin", "Samuel", "Raymond", "Gregory", "Frank",
    "Alexander", "Patrick", "Jack", "Dennis", "Jerry", "Tyler", "Aaron", "Jose",
    "Adam", "Nathan", "Henry", "Douglas", "Zachary", "Peter", "Kyle"
]

FEMALE_NAMES = [
    "Mary", "Patricia", "Jennifer", "Linda", "Barbara", "Elizabeth", "Susan",
    "Jessica", "Sarah", "Karen", "Lisa", "Nancy", "Betty", "Margaret", "Sandra",
    "Ashley", "Kimberly", "Emily", "Donna", "Michelle", "Dorothy", "Carol",
    "Amanda", "Melissa", "Deborah", "Stephanie", "Rebecca", "Sharon", "Laura",
    "Cynthia", "Kathleen", "Amy", "Angela", "Shirley", "Anna", "Brenda",
    "Pamela", "Emma", "Nicole", "Helen", "Samantha", "Katherine", "Christine",
    "Debra", "Rachel", "Carolyn", "Janet", "Catherine", "Maria", "Heather",
    "Diane", "Ruth", "Julie", "Olivia", "Joyce", "Virginia", "Victoria"
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
    "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
    "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson",
    "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson",
    "Walker", "Young", "Allen", "King", "Wright", "Scott", "Torres", "Nguyen",
    "Hill", "Flores", "Green", "Adams", "Nelson", "Baker", "Hall", "Rivera",
    "Campbell", "Mitchell", "Carter", "Roberts", "Turner", "Phillips", "Evans",
    "Parker", "Edwards", "Collins", "Stewart", "Morris", "Murphy", "Cook"
]

EMAIL_DOMAINS = [
    "gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "icloud.com",
    "aol.com", "mail.com", "protonmail.com", "live.com", "msn.com"
]


def generate_email(first_name: str, last_name: str) -> str:
    """Generate a realistic email address."""
    domain = random.choice(EMAIL_DOMAINS)

    # Different email patterns
    patterns = [
        f"{first_name.lower()}.{last_name.lower()}@{domain}",
        f"{first_name.lower()}{last_name.lower()}@{domain}",
        f"{first_name.lower()}_{last_name.lower()}@{domain}",
        f"{first_name[0].lower()}{last_name.lower()}@{domain}",
        f"{first_name.lower()}{last_name[0].lower()}@{domain}",
        f"{first_name.lower()}.{last_name.lower()}{random.randint(1, 99)}@{domain}",
        f"{first_name.lower()}{random.randint(1, 999)}@{domain}",
    ]

    return random.choice(patterns)


def generate_phone() -> str:
    """Generate a realistic US phone number."""
    # Common US area codes
    area_codes = [
        "212", "213", "312", "415", "617", "702", "713", "720", "786", "818",
        "202", "214", "310", "404", "469", "512", "619", "650", "704", "832"
    ]
    area_code = random.choice(area_codes)
    exchange = random.randint(200, 999)
    subscriber = random.randint(1000, 9999)

    return f"+1-{area_code}-{exchange}-{subscriber}"


def add_personal_info(input_path: str, output_path: str):
    """Add personal information columns to the dataset."""
    print(f"Reading dataset from {input_path}...")
    df = pd.read_csv(input_path)

    # Remove trailing empty column if exists
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    print(f"Dataset has {len(df)} rows")

    # Generate personal info based on gender
    first_names = []
    last_names = []
    emails = []
    phones = []

    print("Generating personal information...")
    for idx, row in df.iterrows():
        # Get gender
        gender = row['Gender']

        # Pick appropriate first name
        if gender == 'M':
            first_name = random.choice(MALE_NAMES)
        else:
            first_name = random.choice(FEMALE_NAMES)

        last_name = random.choice(LAST_NAMES)
        email = generate_email(first_name, last_name)
        phone = generate_phone()

        first_names.append(first_name)
        last_names.append(last_name)
        emails.append(email)
        phones.append(phone)

        if (idx + 1) % 2000 == 0:
            print(f"  Processed {idx + 1} rows...")

    # Insert new columns after CLIENTNUM
    df.insert(1, 'First_Name', first_names)
    df.insert(2, 'Last_Name', last_names)
    df.insert(3, 'Email', emails)
    df.insert(4, 'Phone_Number', phones)

    # Save updated dataset
    print(f"Saving updated dataset to {output_path}...")
    df.to_csv(output_path, index=False)

    print("\nDone! Sample of new data:")
    print(df[['CLIENTNUM', 'First_Name', 'Last_Name', 'Email', 'Phone_Number', 'Gender']].head(10))

    return df


if __name__ == "__main__":
    input_file = "/Users/jihane/Desktop/ML2/data/churn2.csv"
    output_file = "/Users/jihane/Desktop/ML2/data/churn2.csv"  # Overwrite original

    # Create backup first
    import shutil
    backup_file = "/Users/jihane/Desktop/ML2/data/churn2_backup.csv"
    shutil.copy(input_file, backup_file)
    print(f"Backup created at {backup_file}")

    add_personal_info(input_file, output_file)
