import pandas as pd
import numpy as np
from faker import Faker
import random

fake = Faker()


def generate_fake_transactions(num_samples=5000):
    transactions = []

    # Original categories list
    original_categories = ['grocery_pos', 'entertainment', 'shopping_pos', 'misc_pos',
                           'shopping_net', 'gas_transport', 'misc_net', 'grocery_net',
                           'food_dining', 'health_fitness', 'kids_pets', 'home',
                           'personal_care', 'travel']

    for _ in range(num_samples):
        transaction = {
            "trans_date_trans_time": fake.date_time_this_decade(),
            "merchant": fake.company(),
            "category": random.choice(original_categories),  # Updated categories
            "amt": round(random.uniform(1, 5000), 2),
            "city": fake.city(),
            "state": fake.state(),
            "lat": round(fake.latitude(), 6),
            "long": round(fake.longitude(), 6),
            "city_pop": random.randint(1000, 1000000),
            "job": fake.job(),
            "dob": fake.date_of_birth(minimum_age=18, maximum_age=80),
            "trans_num": fake.uuid4(),
            "merch_lat": round(fake.latitude(), 6),
            "merch_long": round(fake.longitude(), 6),
            "is_fraud": random.choices([0, 1], weights=[98, 2])[0]  # 2% fraud rate
        }
        transactions.append(transaction)

    df_fake = pd.DataFrame(transactions)

    # Save to CSV
    df_fake.to_csv("fake_transactions.csv", index=False)

    return df_fake


if __name__ == "__main__":
    df = generate_fake_transactions(5000)
    print("✅ Generated fake transactions!")
# import pandas as pd
#
# # Load your fake transactions dataset
# df_fake = pd.read_csv("fake_transactions.csv")
#
# # Assuming 'is_fraud' is the column indicating fraud transactions (1 for fraud, 0 for non-fraud)
# fraud_count = df_fake['is_fraud'].value_counts()
#
# print(fraud_count)