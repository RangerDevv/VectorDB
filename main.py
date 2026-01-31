import vectorDB
data = [
    {"text": "The sky is blue and vast."},
    {"text": "An orange is a type of citrus fruit."},
    {"text": "Formula 1 is the pinnacle of motorsport."},
    {"text": "The ocean covers most of the Earth's surface."},
    {"text": "Apples are a popular and healthy snack."},
    {"text": "Mountains are majestic landforms that rise prominently above their surroundings."},
    {"text": "Soccer is the most popular sport worldwide."},
    {"text": "Bananas are rich in potassium and fiber."},
    {"text": "The desert is characterized by its arid conditions and sparse vegetation."},
    {"text": "Tennis is played on various surfaces including clay, grass, and hard courts."},
    {"text": "Grapes can be eaten fresh or used to make wine."},
    {"text": "Rivers are natural flowing watercourses that usually lead to an ocean, sea, or lake."},
    {"text": "Basketball is a fast-paced game played with a hoop and a ball."},
    {"text": "Cherries are small, round fruits that can be sweet or tart."},
    {"text": "Forests are dense collections of trees and undergrowth."},
    {"text": "Baseball is known as America's pastime."},
    {"text": "Pineapples have a unique sweet and tangy flavor."},
    {"text": "Volcanoes are openings in the Earth's crust that allow molten rock to escape."},
    {"text": "Hockey is played on ice with skates and a puck."},
    {"text": "Strawberries are bright red fruits that are often used in desserts."},
    {"text": "Canyons are deep gorges typically carved by rivers over time."}
]

userQuery = input("Enter your query: ")

vectorDB_instance = vectorDB.vectorDB(data)
lsh_result = vectorDB_instance.search(userQuery, k=5)
print("\nLSH Search Result (Top-k by Hamming distance):")
if lsh_result:
    print(f"Best Match: {lsh_result['best_match']}")
    print(f"Best Match Hamming Distance: {lsh_result['hamming_distance']}")

    print("\nTop matches:")
    for match in lsh_result["matches"]:
        print(f"- Text: {match['text']}")
        print(f"  Hamming Distance: {match['hamming_distance']}")