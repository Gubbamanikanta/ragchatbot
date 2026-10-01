"""
Medical Knowledge Base, NLP Intent Engine, Symptom Severity Classifier,
Emergency Red-Flag Detector, and Multi-language Support for SymptoTrackAI.
"""

import re

# ==========================================
# 1. CLINICAL KNOWLEDGE BASE (24 DISEASES)
# ==========================================

DISEASE_KB = {
    "Acne": {
        "name": "Acne Vulgaris",
        "severity": "MILD",
        "overview": "A common inflammatory skin condition that occurs when hair follicles become clogged with oil and dead skin cells, causing pimples, blackheads, or whiteheads.",
        "self_care": [
            "Wash affected areas twice daily with a gentle, non-abrasive cleanser.",
            "Avoid squeezing, picking, or scrubbing blemishes to prevent scarring.",
            "Use non-comedogenic (oil-free) moisturizers and sunscreen.",
            "Topical treatments containing benzoyl peroxide, salicylic acid, or retinoids may assist."
        ],
        "when_to_see_doctor": "Consult a dermatologist if over-the-counter products do not improve lesions after 6-8 weeks, or if you develop deep cystic or painful nodules.",
        "trusted_source": "Mayo Clinic - Acne Guide",
        "source_url": "https://www.mayoclinic.org/diseases-conditions/acne/symptoms-causes/syc-20368047"
    },
    "Arthritis": {
        "name": "Arthritis / Joint Inflammation",
        "severity": "MODERATE",
        "overview": "Inflammation or degeneration of one or more joints causing stiffness, localized swelling, tenderness, and restricted mobility.",
        "self_care": [
            "Apply heat packs to soothe stiff joints or cold packs for acute swelling.",
            "Engage in low-impact physical activities like swimming, walking, or gentle cycling.",
            "Maintain a healthy body weight to reduce joint load.",
            "Over-the-counter NSAIDs (e.g., ibuprofen) may temporarily relieve mild discomfort under physician guidance."
        ],
        "when_to_see_doctor": "Seek clinical evaluation if joint pain is accompanied by warmth, redness, significant fever, or sudden inability to bear weight.",
        "trusted_source": "CDC - Arthritis Basics",
        "source_url": "https://www.cdc.gov/arthritis/basics/index.html"
    },
    "Bronchial Asthma": {
        "name": "Bronchial Asthma",
        "severity": "URGENT",
        "overview": "A chronic respiratory disorder where airways become inflamed, narrowed, and swollen with extra mucus, causing wheezing, coughing, and dyspnea.",
        "self_care": [
            "Always keep prescribed rescue inhalers (e.g., albuterol) immediately accessible.",
            "Identify and avoid known triggers such as pollen, cold dry air, smoke, or pet dander.",
            "Use an air purifier indoors and monitor peak flow readings if recommended.",
            "Follow your written Asthma Action Plan provided by your physician."
        ],
        "when_to_see_doctor": "EMERGENCY: Seek immediate medical care if rescue inhaler does not bring relief within 15 minutes, if speaking in full sentences is difficult, or if lips/fingernails turn pale or blue.",
        "trusted_source": "WHO - Asthma Fact Sheet",
        "source_url": "https://www.who.int/news-room/fact-sheets/detail/asthma"
    },
    "Cervical spondylosis": {
        "name": "Cervical Spondylosis (Neck Arthritis)",
        "severity": "MODERATE",
        "overview": "Age-related wear and tear affecting spinal disks and vertebrae in your neck, often causing stiffness, neck pain, and muscle spasms.",
        "self_care": [
            "Practice ergonomic posture while working at desks and computer screens.",
            "Do gentle neck stretches and strengthening exercises regularly.",
            "Use a supportive orthopedic neck pillow while sleeping.",
            "Apply alternating warm compresses and ice packs to relieve muscle tension."
        ],
        "when_to_see_doctor": "See a physician if you experience numbness, tingling, or weakness radiating into your shoulders, arms, or hands.",
        "trusted_source": "Mayo Clinic - Cervical Spondylosis",
        "source_url": "https://www.mayoclinic.org/diseases-conditions/cervical-spondylosis/symptoms-causes/syc-20370787"
    },
    "Chicken pox": {
        "name": "Chickenpox (Varicella)",
        "severity": "MODERATE",
        "overview": "A highly contagious viral infection caused by the varicella-zoster virus, characterized by an itchy, blister-like rash across the body.",
        "self_care": [
            "Isolate at home until all blisters have fully crusted over to prevent transmission.",
            "Apply calamine lotion or take lukewarm oatmeal baths to alleviate intense itching.",
            "Keep fingernails trimmed short to prevent secondary bacterial skin infections from scratching.",
            "Drink plenty of fluids and take paracetamol for fever (never give aspirin to children)."
        ],
        "when_to_see_doctor": "Seek medical evaluation if rash spreads to the eyes, if the skin becomes extremely red, swollen or leaks pus, or if shortness of breath develops.",
        "trusted_source": "CDC - Chickenpox (Varicella)",
        "source_url": "https://www.cdc.gov/chickenpox/index.html"
    },
    "Common Cold": {
        "name": "Common Cold / Viral Upper Respiratory Infection",
        "severity": "MILD",
        "overview": "A viral infection of the upper respiratory tract commonly causing a runny or stuffy nose, sneezing, mild sore throat, cough, and low-grade fatigue.",
        "self_care": [
            "Get adequate bed rest and drink plenty of warm fluids (water, broth, herbal tea).",
            "Use saline nasal sprays or steam inhalation to relieve nasal congestion.",
            "Gargle with warm salt water to soothe a scratchy or irritated throat.",
            "Use a clean room humidifier to keep nasal and respiratory passages moist."
        ],
        "when_to_see_doctor": "Consult a healthcare professional if symptoms worsen after 10 days, if fever exceeds 102°F (38.9°C), or if you develop chest pain or wheezing.",
        "trusted_source": "CDC - Common Cold Prevention & Care",
        "source_url": "https://www.cdc.gov/antibiotic-use/common-cold.html"
    },
    "Dengue": {
        "name": "Dengue Fever",
        "severity": "URGENT",
        "overview": "A mosquito-borne viral disease causing sudden high fever, retro-orbital (behind the eyes) eye pain, severe joint/muscle aches ('breakbone fever'), and skin rash.",
        "self_care": [
            "Strict bed rest and aggressive oral hydration with oral rehydration salts (ORS), coconut water, and juices.",
            "Take paracetamol for fever and body aches.",
            "DO NOT take aspirin, ibuprofen, or other NSAIDs, as they significantly increase bleeding risk.",
            "Use mosquito nets and repellents to prevent secondary transmission."
        ],
        "when_to_see_doctor": "URGENT: Go to an emergency room immediately if you experience persistent vomiting, severe abdominal pain, bleeding gums/nose, extreme lethargy, or restlessness.",
        "trusted_source": "WHO - Dengue and Severe Dengue",
        "source_url": "https://www.who.int/news-room/fact-sheets/detail/dengue-and-severe-dengue"
    },
    "Dimorphic Hemorrhoids": {
        "name": "Hemorrhoids (Piles)",
        "severity": "MILD",
        "overview": "Swollen veins located around the lower rectum and anus that can cause discomfort, itching, pain, and occasional bright red bleeding during bowel movements.",
        "self_care": [
            "Increase dietary fiber intake (whole grains, beans, fruits, vegetables) and drink 2-3 liters of water daily.",
            "Take warm sitz baths (soaking the rectal area in warm water) for 15-20 minutes several times a day.",
            "Avoid straining or sitting on the toilet for prolonged periods.",
            "Use mild over-the-counter hemorrhoidal wipes or hydrocortisone ointments as directed."
        ],
        "when_to_see_doctor": "Consult a physician if bleeding is heavy, accompanied by dizziness, or if symptoms persist beyond one week.",
        "trusted_source": "Mayo Clinic - Hemorrhoids Guide",
        "source_url": "https://www.mayoclinic.org/diseases-conditions/hemorrhoids/symptoms-causes/syc-20360268"
    },
    "Fungal infection": {
        "name": "Superficial Fungal Infection (Tinea / Ringworm)",
        "severity": "MILD",
        "overview": "A common fungal skin infection that causes an itchy, circular or scaly rash in warm, moist areas of the body (e.g., athlete's foot, jock itch, body ringworm).",
        "self_care": [
            "Keep the affected skin thoroughly clean, dry, and exposed to fresh air when possible.",
            "Apply over-the-counter topical antifungal creams (e.g., clotrimazole, terbinafine) as directed.",
            "Wear loose, breathable cotton clothing and change damp socks/underwear promptly.",
            "Avoid sharing personal items like towels, bedding, or clothing with others."
        ],
        "when_to_see_doctor": "See a doctor if the infection spreads rapidly, fails to improve after 2 weeks of topical antifungal treatment, or affects the scalp or nails.",
        "trusted_source": "CDC - Fungal Diseases / Ringworm",
        "source_url": "https://www.cdc.gov/fungal/diseases/ringworm/index.html"
    },
    "Hypertension": {
        "name": "Hypertension (High Blood Pressure)",
        "severity": "URGENT",
        "overview": "A chronic cardiovascular condition where the pressure of blood pushing against the arterial walls is persistently elevated, increasing risks of heart attack and stroke.",
        "self_care": [
            "Follow the DASH diet: reduce sodium (salt) intake and consume more potassium-rich vegetables and fruits.",
            "Maintain regular moderate physical exercise (e.g., 30 minutes of brisk walking 5 days a week).",
            "Limit alcohol intake, stop smoking, and manage psychological stress through relaxation techniques.",
            "Regularly monitor and log your blood pressure readings at home."
        ],
        "when_to_see_doctor": "HYPERTENSIVE CRISIS: If blood pressure exceeds 180/120 mmHg, or if high readings occur alongside chest pain, blurred vision, severe headache, or shortness of breath, call emergency services immediately.",
        "trusted_source": "WHO - Hypertension Fact Sheet",
        "source_url": "https://www.who.int/news-room/fact-sheets/detail/hypertension"
    },
    "Impetigo": {
        "name": "Impetigo (Bacterial Skin Infection)",
        "severity": "MODERATE",
        "overview": "A contagious bacterial skin infection, predominantly caused by Staphylococcus or Streptococcus, presenting as red sores that rupture and form honey-colored crusts.",
        "self_care": [
            "Gently wash sores with warm water and antibacterial soap, removing dried crusts if loose.",
            "Apply prescribed antibiotic ointment (e.g., mupirocin) directly to lesions.",
            "Cover sores loosely with clean gauze bandages to prevent scratching and bacterial spread.",
            "Wash hands thoroughly and do not share towels, washcloths, or razors."
        ],
        "when_to_see_doctor": "Consult a physician for prescription antibiotic therapy and if sores become deeply ulcerated, painful, or accompanied by fever.",
        "trusted_source": "Mayo Clinic - Impetigo Diagnosis and Care",
        "source_url": "https://www.mayoclinic.org/diseases-conditions/impetigo/symptoms-causes/syc-20352352"
    },
    "Jaundice": {
        "name": "Jaundice / Hepatic Dysfunction",
        "severity": "URGENT",
        "overview": "Yellowing of the skin and whites of the eyes (sclera) resulting from elevated bilirubin levels in the bloodstream, typically indicating liver, gallbladder, or hemolytic issues.",
        "self_care": [
            "Strictly avoid all alcohol consumption and hepatotoxic medications (including excessive acetaminophen).",
            "Eat a bland, low-fat, nutrient-rich diet with easily digestible carbohydrates and adequate hydration.",
            "Prioritize rest and avoid strenuous physical exertion.",
            "Follow physician orders for liver function blood panels and diagnostic imaging."
        ],
        "when_to_see_doctor": "URGENT: Jaundice always warrants prompt medical diagnosis to identify the underlying cause (hepatitis, gallstones, obstruction, etc.).",
        "trusted_source": "NHS - Jaundice in Adults",
        "source_url": "https://www.nhs.uk/conditions/jaundice/"
    },
    "Malaria": {
        "name": "Malaria",
        "severity": "URGENT",
        "overview": "A life-threatening parasitic infection transmitted by female Anopheles mosquitoes, characterized by cyclical episodes of shaking chills, high fever, sweats, and profound fatigue.",
        "self_care": [
            "Malaria requires prescription antimalarial medication (e.g., Artemisinin-based combination therapies) immediately upon laboratory confirmation.",
            "Maintain high fluid intake to replace fluids lost through high fever and sweating.",
            "Take prescribed antipyretics to help control fever spikes.",
            "Rest completely in a cool, screened room."
        ],
        "when_to_see_doctor": "EMERGENCY: Seek immediate hospital care for diagnostic blood smear / rapid antigen test. Delayed treatment can lead to severe cerebral malaria, renal failure, or death.",
        "trusted_source": "WHO - Malaria Overview",
        "source_url": "https://www.who.int/news-room/fact-sheets/detail/malaria"
    },
    "Migraine": {
        "name": "Migraine Headache",
        "severity": "MODERATE",
        "overview": "A neurological condition typically causing severe throbbing, unilateral headache attacks often accompanied by nausea, vomiting, and hypersensitivity to light and sound.",
        "self_care": [
            "Rest in a quiet, dark, temperature-controlled room during acute headache onset.",
            "Place a cool ice pack or cold damp cloth across the forehead or back of the neck.",
            "Stay well-hydrated and avoid common dietary triggers (aged cheeses, MSG, nitrates, artificial sweeteners).",
            "Keep a consistent sleep schedule and manage daily stress levels."
        ],
        "when_to_see_doctor": "Seek immediate medical attention if you experience a sudden 'thunderclap' headache, confusion, seizure, or neurological deficits (weakness, numbness).",
        "trusted_source": "Mayo Clinic - Migraine Overview",
        "source_url": "https://www.mayoclinic.org/diseases-conditions/migraine-headache/symptoms-causes/syc-20360201"
    },
    "Pneumonia": {
        "name": "Pneumonia (Lung Infection)",
        "severity": "URGENT",
        "overview": "An infection that inflames the air sacs (alveoli) in one or both lungs, which may fill with fluid or purulent material, causing cough with phlegm, fever, chills, and labored breathing.",
        "self_care": [
            "Take all prescribed antibiotics or antivirals exactly as prescribed for the full course.",
            "Get maximal bed rest and avoid all smoke, dust, and environmental irritants.",
            "Drink plenty of fluids to help loosen thick respiratory secretions.",
            "Use a cool-mist humidifier in the bedroom."
        ],
        "when_to_see_doctor": "EMERGENCY: Seek urgent medical care if you experience shortness of breath, sharp pleuritic chest pain that worsens with breathing, bluish discoloration of the lips, or high fever with confusion.",
        "trusted_source": "CDC - Pneumonia Facts & Prevention",
        "source_url": "https://www.cdc.gov/pneumonia/index.html"
    },
    "Psoriasis": {
        "name": "Psoriasis (Autoimmune Skin Disorder)",
        "severity": "MODERATE",
        "overview": "A chronic autoimmune dermatological condition characterized by rapid skin cell turnover leading to thick, red patches covered with silvery scales, commonly on elbows, knees, scalp, and torso.",
        "self_care": [
            "Apply thick emollient moisturizers or petroleum jelly daily to prevent skin drying and cracking.",
            "Take short lukewarm baths with bath oils or colloidal oatmeal; avoid harsh hot water.",
            "Expose skin to controlled, brief amounts of natural sunlight under medical guidance.",
            "Manage stress and avoid skin trauma (cuts, friction, severe sunburns)."
        ],
        "when_to_see_doctor": "Consult a dermatologist for specialized topical corticosteroids, phototherapy, or systemic biologic therapies, especially if joints become swollen or painful.",
        "trusted_source": "Mayo Clinic - Psoriasis Overview",
        "source_url": "https://www.mayoclinic.org/diseases-conditions/psoriasis/symptoms-causes/syc-20355840"
    },
    "Typhoid": {
        "name": "Typhoid Fever",
        "severity": "URGENT",
        "overview": "A systemic bacterial infection caused by Salmonella Typhi, spread through contaminated water or food, presenting with step-ladder high fever, abdominal pain, rose spots, and headache.",
        "self_care": [
            "Requires immediate prompt medical diagnosis and prescription antibiotic treatment.",
            "Maintain strict hydration with bottled, boiled, or purified water and oral rehydration salts.",
            "Eat soft, easily digestible foods (rice porridge, bananas, boiled potatoes).",
            "Wash hands thoroughly with soap after using the restroom to avoid spreading the bacteria."
        ],
        "when_to_see_doctor": "URGENT: Requires medical supervision for proper blood culture and antibiotics. Seek hospital care if experiencing severe abdominal pain, persistent vomiting, or delirium.",
        "trusted_source": "WHO - Typhoid Fact Sheet",
        "source_url": "https://www.who.int/news-room/fact-sheets/detail/typhoid"
    },
    "Varicose Veins": {
        "name": "Varicose Veins",
        "severity": "MILD",
        "overview": "Enlarged, twisted, and swollen veins, usually occurring in the legs and feet, caused by weakened or damaged venous valves allowing blood to pool.",
        "self_care": [
            "Wear gradient compression stockings as recommended by a vascular specialist.",
            "Elevate your legs above heart level several times a day for 15 minutes.",
            "Avoid prolonged uninterrupted periods of standing or sitting.",
            "Exercise regularly to improve venous circulation in the calf muscles."
        ],
        "when_to_see_doctor": "Seek clinical evaluation if you experience skin discoloration, hardening, severe localized leg pain, bleeding veins, or leg ulcers.",
        "trusted_source": "Mayo Clinic - Varicose Veins Guide",
        "source_url": "https://www.mayoclinic.org/diseases-conditions/varicose-veins/symptoms-causes/syc-20350656"
    },
    "allergy": {
        "name": "Allergic Reaction / Hypersensitivity",
        "severity": "MODERATE",
        "overview": "An immune system overreaction to typically harmless foreign substances (allergens) such as pollen, pet dander, mold, dust mites, insect stings, or certain foods.",
        "self_care": [
            "Identify and strictly avoid exposure to the causative allergen whenever possible.",
            "Over-the-counter non-drowsy antihistamines (e.g., cetirizine, fexofenadine) can relieve mild symptoms.",
            "Use saline nasal irrigation to clear environmental allergens from nasal passages.",
            "Apply cool compresses to itchy skin or hives."
        ],
        "when_to_see_doctor": "RED FLAG / ANAPHYLAXIS: Seek emergency medical care (call 112/911/108) immediately if you experience swelling of the tongue/throat, difficulty breathing, throat tightness, or sudden dizziness.",
        "trusted_source": "CDC - Allergies Overview",
        "source_url": "https://www.cdc.gov/climateandhealth/effects/allergy.htm"
    },
    "diabetes": {
        "name": "Diabetes Mellitus",
        "severity": "MODERATE",
        "overview": "A chronic metabolic disease characterized by elevated blood glucose levels, resulting from insufficient insulin production or cellular insulin resistance, presenting with frequent urination and excessive thirst.",
        "self_care": [
            "Follow a balanced medical nutritional plan rich in whole grains, fiber, and lean proteins while limiting refined sugars.",
            "Engage in regular aerobic and resistance exercise to improve insulin sensitivity.",
            "Routinely monitor blood sugar (glucose) levels using a glucometer and record readings.",
            "Inspect your feet daily for small cuts, blisters, or sores."
        ],
        "when_to_see_doctor": "Seek emergency medical attention if blood glucose is critically high or low, or if you develop fruity-scented breath, confusion, or extreme vomiting (diabetic ketoacidosis).",
        "trusted_source": "WHO - Diabetes Facts",
        "source_url": "https://www.who.int/news-room/fact-sheets/detail/diabetes"
    },
    "drug reaction": {
        "name": "Adverse Drug Reaction",
        "severity": "URGENT",
        "overview": "An unwanted or harmful effect experienced after taking a pharmaceutical medication, ranging from mild skin rash or nausea to severe hypersensitivity reactions.",
        "self_care": [
            "Immediately contact your prescribing physician or pharmacist to discuss the reaction before stopping essential medications.",
            "Document the exact medication name, dosage, time taken, and specific symptoms that appeared.",
            "Avoid taking further doses of the suspected offending agent unless advised by a doctor.",
            "Stay hydrated and avoid applying unprescribed chemical ointments to drug rashes."
        ],
        "when_to_see_doctor": "EMERGENCY: Call emergency services immediately if the reaction involves facial swelling, blistering/peeling skin (Stevens-Johnson syndrome), high fever, or difficulty breathing.",
        "trusted_source": "Mayo Clinic - Drug Allergy",
        "source_url": "https://www.mayoclinic.org/diseases-conditions/drug-allergy/symptoms-causes/syc-20371835"
    },
    "gastroesophageal reflux disease": {
        "name": "Gastroesophageal Reflux Disease (GERD)",
        "severity": "MODERATE",
        "overview": "A chronic digestive disorder where stomach acid or bile irritates the lining of the food pipe (esophagus), causing heartburn, acid regurgitation, and chest discomfort.",
        "self_care": [
            "Eat smaller, more frequent meals rather than large, heavy dinners.",
            "Avoid trigger foods: citrus fruits, tomatoes, caffeine, chocolate, peppermint, and spicy or fatty meals.",
            "Do not lie down for at least 2 to 3 hours after eating.",
            "Elevate the head of your bed by 6 inches using bed risers or an acid-reflux wedge pillow."
        ],
        "when_to_see_doctor": "Consult a physician if heartburn occurs more than twice a week, if swallowing becomes painful or difficult, or if accompanied by unexplained weight loss.",
        "trusted_source": "Mayo Clinic - GERD Guide",
        "source_url": "https://www.mayoclinic.org/diseases-conditions/gerd/symptoms-causes/syc-20361940"
    },
    "peptic ulcer disease": {
        "name": "Peptic Ulcer Disease",
        "severity": "MODERATE",
        "overview": "Sores that develop on the inner lining of your stomach and the upper portion of your small intestine (duodenum), typically caused by H. pylori infection or prolonged use of NSAID painkillers.",
        "self_care": [
            "Discontinue or minimize use of NSAIDs (e.g., ibuprofen, naproxen, aspirin) after consulting your physician.",
            "Avoid alcohol, smoking, and highly acidic or heavily spiced foods that irritate the stomach lining.",
            "Eat smaller, balanced meals at regular times to prevent prolonged empty stomach acidity.",
            "Take prescribed acid reducers (e.g., PPIs or H2 blockers) as instructed by your doctor."
        ],
        "when_to_see_doctor": "EMERGENCY: Seek immediate hospital care if you vomit red blood or material that looks like dark coffee grounds, or if you pass black, tarry stools.",
        "trusted_source": "Mayo Clinic - Peptic Ulcer Overview",
        "source_url": "https://www.mayoclinic.org/diseases-conditions/peptic-ulcer/symptoms-causes/syc-20354223"
    },
    "urinary tract infection": {
        "name": "Urinary Tract Infection (UTI)",
        "severity": "MODERATE",
        "overview": "An infection in any part of the urinary system (kidneys, ureters, bladder, urethra), most commonly causing painful, burning urination, pelvic cramping, and frequent urge to urinate.",
        "self_care": [
            "Drink plenty of water throughout the day to flush bacteria from the urinary tract.",
            "Urinate promptly when the urge arises; avoid holding in urine for extended periods.",
            "Wipe from front to back after using the toilet to prevent bacterial transfer from the anal region.",
            "Avoid irritating feminine hygiene sprays, douches, and scented bath products."
        ],
        "when_to_see_doctor": "See a doctor promptly for prescription antibiotics. Seek immediate urgent care if you develop high fever, shaking chills, nausea, or flank/back pain (signs of kidney infection).",
        "trusted_source": "CDC - Urinary Tract Infection Basics",
        "source_url": "https://www.cdc.gov/antibiotic-use/uti.html"
    }
}


# ==========================================
# 2. EMERGENCY RED-FLAG PATTERNS
# ==========================================

RED_FLAG_PATTERNS = [
    r"\b(chest\s*pain|chest\s*tightness|chest\s*pressure|pain\s*in\s*chest)\b",
    r"\b(shortness\s*of\s*breath|difficulty\s*breathing|trouble\s*breathing|cant\s*breathe|cannot\s*breathe|gasping)\b",
    r"\b(heart\s*attack|stroke|paralysis|facial\s*droop|slurred\s*speech|sudden\s*weakness)\b",
    r"\b(coughing\s*blood|vomiting\s*blood|blood\s*in\s*vomit|severe\s*bleeding|hemorrhage)\b",
    r"\b(unconscious|fainting|passed\s*out|blackout|seizure|convulsions)\b",
    r"\b(anaphylaxis|throat\s*swelling|tongue\s*swelling|lips\s*swollen|unable\s*to\s*swallow)\b",
    r"\b(stiff\s*neck\s*with\s*fever|neck\s*stiffness\s*high\s*fever)\b",
]

def check_red_flags(text):
    """Returns True if input text matches critical clinical emergency patterns."""
    text_clean = text.lower()
    for pattern in RED_FLAG_PATTERNS:
        if re.search(pattern, text_clean):
            return True
    return False


# ==========================================
# 3. NLP INTENT RECOGNITION
# ==========================================

GREETING_PATTERNS = [
    r"^(hi|hello|hey|greetings|howdy|holla|namaste|namaskar|salaam|ssup|yo)\b",
    r"\b(good\s*(morning|afternoon|evening|day|mrg|mrng|noon))\b",
    r"^hi[,.\s]+good\s*(mrg|mrng|morning|day)",
    r"\b(how\s*are\s*you|how\s*r\s*u|how\s*are\s*things)\b"
]

BOT_INFO_PATTERNS = [
    r"\b(who\s*are\s*you|what\s*are\s*you|what\s*can\s*you\s*do|tell\s*me\s*about\s*yourself|help\s*me|help)\b"
]

GRATITUDE_PATTERNS = [
    r"\b(thank\s*you|thanks|thx|thankyou|appreciate\s*it|great\s*help)\b"
]

FAREWELL_PATTERNS = [
    r"\b(bye|goodbye|see\s*you|take\s*care|cya|exit|quit)\b"
]

MEDICAL_KEYWORD_PATTERNS = [
    r"\b(pain|ache|fever|cough|cold|headache|rash|itching|itch|skin|swell|stomach|throat|vomit|nausea|dizzy|fatigue|tired|weak|breath|bleed|lesion|blister|joint|peel|allergy|chill|burn|chest|abdomen|back|spot|redness|sore|ulcer|muscle|cramp|diarrhea|constipation|infection)\b"
]

def detect_intent(text):
    """
    Classifies user intent using rule-based NLP:
    - 'EMERGENCY': Urgent life-threatening symptoms
    - 'GREETING': Casual greetings / morning greetings
    - 'BOT_INFO': Questions about the chatbot
    - 'GRATITUDE': Thanks / appreciation
    - 'FAREWELL': Goodbye
    - 'SYMPTOM_QUERY': User reporting medical symptoms
    - 'UNCLEAR': General non-medical text without recognizable symptoms
    """
    cleaned = text.strip().lower()

    if check_red_flags(cleaned):
        return "EMERGENCY"

    for pattern in GREETING_PATTERNS:
        if re.search(pattern, cleaned):
            return "GREETING"

    for pattern in BOT_INFO_PATTERNS:
        if re.search(pattern, cleaned):
            return "BOT_INFO"

    for pattern in GRATITUDE_PATTERNS:
        if re.search(pattern, cleaned):
            return "GRATITUDE"

    for pattern in FAREWELL_PATTERNS:
        if re.search(pattern, cleaned):
            return "FAREWELL"

    for pattern in MEDICAL_KEYWORD_PATTERNS:
        if re.search(pattern, cleaned):
            return "SYMPTOM_QUERY"

    # If length is long (>20 chars), treat as potential symptom query to let vector model evaluate
    if len(cleaned.split()) >= 3:
        return "SYMPTOM_QUERY"

    return "UNCLEAR"


# ==========================================
# 4. CLINICAL FOLLOW-UP QUESTIONS BANK
# ==========================================

def get_follow_up_questions(disease_key, text):
    """
    Generates targeted clinical follow-up questions based on the predicted disease
    and user input to simulate a clinical consultation.
    """
    questions = [
        "How many days have you been experiencing these symptoms (e.g., less than 2 days, 1 week, or longer)?",
        "On a scale of 1 to 10, how severe is your discomfort or pain?",
    ]

    disease_lower = disease_key.lower()

    if any(k in disease_lower for k in ["cold", "asthma", "pneumonia"]):
        questions.append("Are you having any difficulty breathing, wheezing, or chest tightness when resting?")
        questions.append("Have you measured your body temperature, and is there any phlegm/mucus?")
    elif any(k in disease_lower for k in ["dengue", "malaria", "typhoid"]):
        questions.append("Have you noticed severe shivering, body aches, joint pain, or pain behind your eyes?")
        questions.append("Have you recently traveled or noticed mosquito bites in your area?")
    elif any(k in disease_lower for k in ["psoriasis", "acne", "fungal", "impetigo", "chicken pox"]):
        questions.append("Is the skin rash spreading, blistering, peeling, or causing intense itching?")
        questions.append("Have you recently changed any skincare products, soaps, or medications?")
    elif any(k in disease_lower for k in ["gerd", "peptic", "hemorrhoids"]):
        questions.append("Do the symptoms worsen after eating spicy, acidic, or heavy meals, or when lying down?")
        questions.append("Have you experienced any nausea, vomiting, or changes in bowel movements?")
    elif any(k in disease_lower for k in ["arthritis", "spondylosis", "varicose"]):
        questions.append("Does the stiffness or swelling worsen in the morning or after prolonged rest?")
        questions.append("Does walking or gentle movement improve or worsen the discomfort?")
    elif any(k in disease_lower for k in ["migraine", "hypertension"]):
        questions.append("Are you sensitive to bright lights or loud sounds, or experiencing visual disturbances (auras)?")
        questions.append("Have you checked your blood pressure recently?")
    elif "uti" in disease_lower or "urinary" in disease_lower:
        questions.append("Are you experiencing any burning sensation, lower pelvic pain, or increased frequency of urination?")
        questions.append("Have you noticed any fever, lower back pain, or cloudy urine?")
    else:
        questions.append("Do you have any existing chronic conditions (such as diabetes, hypertension, or allergies)?")

    return questions


# ==========================================
# 5. MULTI-LANGUAGE LOCALIZATION
# ==========================================

TRANSLATIONS = {
    "en": {
        "greeting": "Hello! I am SymptoTrackAI, your intelligent health and symptom monitoring assistant. How are you feeling today?\n\nPlease describe any symptoms you are experiencing (for example: 'I have a high fever with shivering', 'red itchy rash on my arms', or 'severe throbbing headache').",
        "bot_info": "I am SymptoTrackAI, an AI-powered symptom monitoring chatbot built with Retrieval-Augmented Generation (RAG) and clinical knowledge bases. I can analyze your symptoms, estimate severity, ask follow-up questions, and provide evidence-based guidance from trusted sources like WHO and Mayo Clinic.",
        "gratitude": "You are very welcome! Take good care of your health. Feel free to describe any other symptoms if you have further concerns.",
        "farewell": "Goodbye! Stay safe and healthy. Remember to consult a qualified physician if your symptoms persist or worsen.",
        "unclear": "I couldn't identify specific medical symptoms in your message. Could you please describe your health concern or symptoms in more detail (e.g., headache, fever, rash, cough, or stomach pain)?",
        "emergency_title": "⚠️ URGENT MEDICAL EMERGENCY ALERT",
        "emergency_msg": "Your message mentions symptoms that may indicate a critical or life-threatening condition (such as cardiac distress, stroke signs, or severe respiratory failure). Please call emergency services (112 / 911 / 108) or proceed to the nearest emergency room immediately!",
        "assessment_title": "Clinical Assessment & Monitoring",
        "predicted_condition": "Predicted Potential Condition",
        "severity_label": "Severity Level",
        "mild": "🟢 Low / Mild (Self-limiting / Home Monitoring)",
        "moderate": "🟡 Moderate (Medical Evaluation Recommended)",
        "urgent": "🔴 Urgent / High (Seek Prompt Medical Care)",
        "overview_label": "Clinical Overview",
        "self_care_label": "Recommended First-Line Care & Precautions",
        "warning_label": "When to Consult a Physician Immediately",
        "followup_label": "Clinical Follow-up Questions",
        "source_label": "Trusted Medical Source",
        "disclaimer": "Medical Disclaimer: SymptoTrackAI is an educational and symptom-monitoring tool and does not provide an official diagnosis or substitute for professional medical advice. Always consult a licensed healthcare provider."
    },
    "hi": {
        "greeting": "नमस्ते! मैं SymptoTrackAI हूँ, आपका AI स्वास्थ्य और लक्षण निगरानी सहायक। आप आज कैसा महसूस कर रहे हैं?\n\nकृपया अपने लक्षणों का वर्णन करें (उदाहरण के लिए: 'मुझे ठंड लगकर तेज बुखार है', 'हाथों पर लाल दाने और खुजली है', या 'सिर में तेज दर्द है')।",
        "bot_info": "मैं SymptoTrackAI हूँ, एक AI-आधारित स्वास्थ्य सहायक। मैं आपके लक्षणों का विश्लेषण कर सकता हूँ, गंभीरता का आकलन कर सकता हूँ और WHO तथा Mayo Clinic जैसे विश्वसनीय स्रोतों से मार्गदर्शन प्रदान कर सकता हूँ।",
        "gratitude": "आपका स्वागत है! अपने स्वास्थ्य का ध्यान रखें। यदि कोई अन्य लक्षण महसूस हो, तो बेझिझक साझा करें।",
        "farewell": "नमस्ते! अपना ख्याल रखें और स्वस्थ रहें। यदि लक्षण बने रहें, तो तुरंत चिकित्सक से संपर्क करें।",
        "unclear": "मैं आपके संदेश में स्पष्ट लक्षण नहीं पहचान पाया। कृपया अपने स्वास्थ्य के लक्षणों का विस्तार से वर्णन करें (जैसे: बुखार, खांसी, सिरदर्द, दाने या पेट दर्द)।",
        "emergency_title": "⚠️ आपातकालीन चिकित्सा चेतावनी (EMERGENCY)",
        "emergency_msg": "आपके संदेश में ऐसे लक्षण पाए गए हैं जो गंभीर स्थिति का संकेत हो सकते हैं। कृपया तुरंत आपातकालीन सेवाओं (112 / 108) पर कॉल करें या नजदीकी अस्पताल के आपातकालीन विभाग में जाएँ!",
        "assessment_title": "चिकित्सीय मूल्यांकन और निगरानी",
        "predicted_condition": "अनुमानित संभावित स्थिति",
        "severity_label": "गंभीरता स्तर",
        "mild": "🟢 हल्का / सामान्य (घर पर निगरानी)",
        "moderate": "🟡 मध्यम (चिकित्सक से परामर्श अनुशंसित)",
        "urgent": "🔴 गंभीर / आपातकालीन (शीघ्र चिकित्सीय परामर्श लें)",
        "overview_label": "चिकित्सीय विवरण",
        "self_care_label": "प्राथमिक देखभाल और सावधानियां",
        "warning_label": "तुरंत डॉक्टर के पास कब जाएँ",
        "followup_label": "अनुवर्ती प्रश्न (Follow-up Questions)",
        "source_label": "विश्वसनीय चिकित्सा स्रोत",
        "disclaimer": "चिकित्सा अस्वीकरण: SymptoTrackAI केवल सूचना और लक्षण निगरानी के लिए है, यह पेशेवर चिकित्सा निदान का विकल्प नहीं है। हमेशा एक योग्य डॉक्टर से परामर्श लें।"
    },
    "te": {
        "greeting": "నమస్కారం! నేను SymptoTrackAI, మీ ఆరోగ్య మరియు లక్షణాల పర్యవేక్షణ AI సహాయకుడిని. ఈరోజు మీ ఆరోగ్యం ఎలా ఉంది?\n\nదయచేసి మీ లక్షణాలను వివరించండి (ఉదాహరణకు: 'నాకు చలితో కూడిన జ్వరం ఉంది', 'చేతులపై ఎర్రటి దద్దుర్లు మరియు దురద', లేదా 'తీవ్రమైన తలనొప్పి').",
        "bot_info": "నేను SymptoTrackAI. నేను మీ లక్షణాలను విశ్లేషించి, తీవ్రతను అంచనా వేసి, WHO మరియు Mayo Clinic వంటి విశ్వసనీయ వైద్య వనరుల ఆధారంగా మార్గదర్శకత్వాన్ని అందిస్తాను.",
        "gratitude": "ధన్యవాదాలు! మీ ఆరోగ్యాన్ని జాగ్రత్తగా చూసుకోండి. ఇంకా ఏవైనా లక్షణాలు ఉంటే సంకోచించకుండా అడగండి.",
        "farewell": "వీడ్కోలు! ఆరోగ్యంగా ఉండండి. లక్షణాలు తగ్గకపోతే వెంటనే వైద్యుడిని సంప్రదించండి.",
        "unclear": "మీ సందేశంలో స్పష్టమైన లక్షణాలు గుర్తించబడలేదు. దయచేసి మీ లక్షణాలను మరింత వివరంగా చెప్పండి (ఉదాహరణకు: జ్వరం, దగ్గు, తలనొప్పి, లేదా కడుపు నొప్పి).",
        "emergency_title": "⚠️ అత్యవసర వైద్య హెచ్చరిక (EMERGENCY)",
        "emergency_msg": "మీరు పేర్కొన్న లక్షణాలు అత్యవసర వైద్య పరిస్థితిని సూచించవచ్చు. దయచేసి వెంటనే అత్యవసర సేవలైన 112 / 108 కి కాల్ చేయండి లేదా సమీపంలోని ఆసుపత్రికి వెళ్లండి!",
        "assessment_title": "వైద్య అంచనా & పర్యవేక్షణ",
        "predicted_condition": "అంచనా వేయబడిన పరిస్థితి",
        "severity_label": "తీవ్రత స్థాయి",
        "mild": "🟢 స్వల్పం / సాధారణం (ఇంటి వద్ద పర్యవేక్షణ)",
        "moderate": "🟡 మధ్యస్థం (వైద్యుడిని సంప్రదించడం మంచిది)",
        "urgent": "🔴 తీవ్రమైనది (వెంటనే వైద్య సహాయం పొందండి)",
        "overview_label": "వివరణ",
        "self_care_label": "స్వీయ సంరక్షణ & జాగ్రత్తలు",
        "warning_label": "వైద్యుడిని ఎప్పుడు సంప్రదించాలి",
        "followup_label": "తదుపరి ప్రశ్నలు (Follow-up Questions)",
        "source_label": "విశ్వసనీయ వైద్య వనరు",
        "disclaimer": "వైద్య గమనిక: SymptoTrackAI కేవలం సమాచార ప్రయోజనాల కోసం మాత్రమే. ఇది వృత్తిపరమైన వైద్య సలహాకు ప్రత్యామ్నాయం కాదు."
    },
    "es": {
        "greeting": "¡Hola! Soy SymptoTrackAI, tu asistente inteligente de salud y monitoreo de síntomas. ¿Cómo te sientes hoy?\n\nPor favor, describe los síntomas que estás experimentando (por ejemplo: 'tengo fiebre alta con escalofríos', 'sarpullido rojo con picazón', o 'dolor de cabeza pulsátil').",
        "bot_info": "Soy SymptoTrackAI, un chatbot médico inteligente para monitoreo de síntomas basado en IA. Analizo síntomas, estimo la gravedad y ofrezco pautas basadas en evidencia de fuentes confiables como la OMS y Mayo Clinic.",
        "gratitude": "¡De nada! Cuida mucho tu salud. No dudes en describir otros síntomas si tienes más dudas.",
        "farewell": "¡Hasta luego! Mantente saludable. Si los síntomas persisten o empeoran, consulta a un médico calificado.",
        "unclear": "No pude identificar síntomas médicos claros en tu mensaje. Por favor describe lo que sientes (por ejemplo: fiebre, tos, dolor de cabeza, sarpullido o dolor estomacal).",
        "emergency_title": "⚠️ ALERTA MÉDICA DE EMERGENCIA",
        "emergency_msg": "Tu mensaje menciona síntomas que pueden indicar una condición médica crítica o potencialmente mortal. ¡Llama a los servicios de emergencia (911 / 112) o acude a la sala de emergencias más cercana de inmediato!",
        "assessment_title": "Evaluación Clínica y Monitoreo",
        "predicted_condition": "Condición Potencial Estimada",
        "severity_label": "Nivel de Gravedad",
        "mild": "🟢 Leve (Monitoreo en el hogar)",
        "moderate": "🟡 Moderado (Se recomienda consulta médica)",
        "urgent": "🔴 Urgente / Grave (Busca atención médica inmediata)",
        "overview_label": "Resumen Clínico",
        "self_care_label": "Cuidados Iniciales y Precauciones",
        "warning_label": "Cuándo Acudir al Médico de Inmediato",
        "followup_label": "Preguntas de Seguimiento",
        "source_label": "Fuente Médica de Confianza",
        "disclaimer": "Descargo de responsabilidad médica: SymptoTrackAI es una herramienta educativa y de monitoreo de síntomas. No sustituye la consulta médica profesional."
    }
}

def detect_language(text):
    """Detects if text is in Hindi (Devanagari), Telugu, Spanish, or English."""
    # Devanagari script: \u0900-\u097F
    if re.search(r"[\u0900-\u097F]", text):
        return "hi"
    # Telugu script: \u0c00-\u0c7F
    if re.search(r"[\u0c00-\u0c7F]", text):
        return "te"
    # Spanish distinctive words or accents
    if re.search(r"\b(hola|gracias|adios|tengo|fiebre|dolor|cabeza|tos|buenos\s*dias)\b|[áéíóúñ¿¡]", text, re.IGNORECASE):
        return "es"
    # Common Hindi in Roman script
    if re.search(r"\b(namaste|bukhar|dard|khansi|kya|hai|kaise|shukriya|alvida)\b", text, re.IGNORECASE):
        return "hi"
    # Common Telugu in Roman script
    if re.search(r"\b(namaskaram|jvaram|noppi|daggu|ela|undi|dhanyavadalu)\b", text, re.IGNORECASE):
        return "te"
    return "en"


def get_localized_response(intent, lang="en"):
    """Returns localized greeting, bot info, gratitude, or unclear message."""
    t = TRANSLATIONS.get(lang, TRANSLATIONS["en"])
    if intent == "GREETING":
        return t["greeting"]
    elif intent == "BOT_INFO":
        return t["bot_info"]
    elif intent == "GRATITUDE":
        return t["gratitude"]
    elif intent == "FAREWELL":
        return t["farewell"]
    else:
        return t["unclear"]
