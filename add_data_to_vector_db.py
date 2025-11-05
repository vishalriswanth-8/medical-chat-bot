# This script demonstrates two methods for adding data to a FAISS vector database.

import os
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- 1. PREPARE COMMON COMPONENTS ---
# Make sure you have `nomic-embed-text` pulled via Ollama.
embeddings = OllamaEmbeddings(model="nomic-embed-text")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
db_path = "faiss_index.faiss"

# This is the new data you want to add to your database.
# Replace this with your actual new medical documents.
new_medical_data = [
    """
COLD:
The common cold is a viral infection of your nose and throat (upper respiratory tract). It's generally harmless, but it's highly contagious and can be uncomfortable. It's typically caused by various viruses, most commonly rhinoviruses.
Common Symptoms
Cold symptoms usually appear a few days after exposure to the virus and can include:
•	Stuffy or runny nose (discharge may start clear and become thicker/colored)
•	Sore or scratchy throat
•	Coughing
•	Sneezing
•	Headache or mild body aches
•	Mild fatigue
•	Sometimes a low-grade fever (more common in children)
•	Query successful
HOME REMEDIES AND SELF-CARE
•	Rest: Get plenty of sleep to help your body recover.
•	Fluids: Drink lots of water, juice, or clear broth to stay hydrated. Warm liquids like decaffeinated tea can be soothing. Avoid alcohol and caffeine.
•	Sore Throat Relief:
       Gargle with warm salt water (1/4 to 1/2 teaspoon of salt in 8 ounces of warm water).
 Use throat lozenges or hard candy (not for young children).
•	Honey can soothe a cough and sore throat (do not give to babies younger than 1 year old).
•	Congestion Relief:
•	Use a humidifier or cool-mist vaporizer to moisten the air.
•	Breathe in steam from a bowl of hot water or a hot shower.
•	Use saline nasal sprays or drops to help loosen mucus.
•	Petroleum jelly can be applied to the skin around your nose to soothe chapping from blowing.

WHEN TO SEE A DOCTOR
Most colds resolve within 7 to 10 days. However, seek medical attention if you or your child have:
•	A fever that lasts longer than 3-4 days.
•	Symptoms that worsen or don't improve after 10 days.
•	Difficulty breathing, shortness of breath, or wheezing.
•	Severe headache or severe sore throat.
•	Symptoms that clear up but then return or get worse (which may indicate a secondary bacterial infection).
MEDICINE:
           I.  For Pain, Aches, and Fever:
1.	Tylenol
2.	Advil, Motrin
           II. For Stuffy Nose and Congestion:
                                            Oral (Pill/Liquid):
1.	Pseudoephedrine
2.	Phenylephrine
                                              Nasal Sprays:
1.Oxymetazoline
2.Saline Nasal Spray
           Multi-Symptom Cold Products
                                 Dayquil, Mucinex Fast-Max Daytime, Tylenol Cold & Flu
                                  NyQuil, Mucinex Nightshift

STOMACH PAIN (ABDOMINAL DISCOMFORT):
Pain relievers that also treat inflammation (like ibuprofen or naproxen) can sometimes worsen stomach pain or ulcers. You should treat the stomach symptom separately, and often non-medication remedies are safer.
•	Antacids/Acid Reducers: If the pain is due to indigestion or acidity (e.g., Tums, Rolaids, Gaviscon or Famotidine/Pepcid).
•	Rehydration: If the stomach pain is due to an infection that also caused vomiting or diarrhoea, oral rehydration solutions (ORS) are crucial (e.g., Electrolyte/ORS packets)
You must consult a doctor or pharmacist, especially if:
•	The stomach pain is severe or localized.
•	You are vomiting or have severe diarrhoea.
•	The fever is very high or lasts for more than 2-3 days.
•	You have any signs of dehydration.
FOR FEVER:
                   General Advice for Fever:
•	Follow Dosing Instructions: Always follow the dosage instructions on the package carefully.
•	Hydration: Drink plenty of fluids (water, clear broths, electrolyte solutions) to prevent dehydration caused by fever and sweating.
•	Rest: Get plenty of rest, as this helps your body fight the infection.
•	Do not give Aspirin to children or teenagers due to the risk of Reye's syndrome.
MEDICINE:
         Acetaminophen
         Non-Steroidal Anti-Inflammatory Drugs (NSAIDs)
                         Ibuprofen
                         Naproxen
Consult a doctor immediately if:
•	The fever is very high (103∘F or higher) or lasts for more than 3 days.
•	The fever is accompanied by severe symptoms like confusion, stiff neck, severe headache, or trouble breathing.
•	You are experiencing severe or persistent stomach pain (as mentioned in your original request).
JOINT PAIN:
       NSAIDs (Non-Steroidal Anti-Inflammatory Drugs):
                   Advil, Motrin:
                              Reduces inflammation and swelling, and relieves pain. Often the first choice when joint pain involves swelling.
Naproxen Sodium :   Aleve( Reduces inflammation and swelling, and relieves pain. It is longer-acting, often taken every 8 to 12 hours.)
   Analgesic:
           Acetaminophen Tylenol, Paracetamol ( Primarily a pain reliever. It does not reduce inflammation. It is generally easier on the stomach than NSAIDs.)

DIABTETICS:
      Diabetes Mellitus (commonly known as diabetes) is a chronic (long-lasting) health condition that affects how your body turns food into energy.
             It is characterized by high blood glucose (sugar) levels, which occurs because the body either doesn't produce enough insulin or doesn't use the insulin it makes effectively (a condition called insulin resistance).
TYPE1 Diabetes :
           An autoimmune reaction causes the immune system to destroy the insulin-producing cells in the pancreas .Often diagnosed in children and young adults, but can occur at any age. Requires daily insulin injections or use of an insulin pump.
Type 2 Diabetes: 
           The body's cells become resistant to insulin, and the pancreas cannot produce enough insulin to compensate. Most common type, often diagnosed in adults, but increasingly seen in children and teens.  Managed with lifestyle changes (diet and exercise), oral medications (like Metformin), and sometimes insulin.
Gestational Diabetes: 
           High blood sugar that develops during pregnancy. It is caused by hormones from the placenta that create insulin resistance .Develops in some pregnant women who have never had diabetes. Usually managed with diet/exercise, but may require insulin. Typically resolves after delivery, but increases the risk of Type 2 diabetes later in life. 
Prediabetes :
Blood sugar levels are higher than normal, but not high enough for a Type 2 diagnosis. Affects millions of adults and is a stage before Type 2 diabetes. Reversible with lifestyle changes like weight loss and increased physical activity.
Common Symptoms
Symptoms can vary depending on the type of diabetes, but the most common ones include:
•	Increased thirst
•	Frequent urination
•	Feeling very hungry (polyphagia)
•	Extreme fatigue
•	Blurred vision
•	Slow-healing sores or frequent infections (skin, gum, vaginal)
•	Unexplained weight loss (more common in Type 1)
•	Tingling, pain, or numbness in the hands or feet (a sign of nerve damage, more common in Type 2)
Long-Term Complications
Over time, consistently high blood sugar can damage nerves and blood vessels throughout the body. The best way to prevent these is through strict blood sugar control.
•	Cardiovascular Disease: Increased risk of heart attack, stroke, and high blood pressure.
•	Nerve Damage (Neuropathy): Can cause tingling, pain, or numbness, typically starting in the feet and hands.
•	Kidney Damage (Nephropathy): Can lead to chronic kidney disease or kidney failure.
•	Eye Damage (Retinopathy): Can lead to vision loss and blindness.
•	Foot Problems: Nerve damage and poor blood flow can lead to slow-healing ulcers and, in severe cases, amputation.
•	Dental Issues: Increased risk of gum disease and other oral health problems.
Management and Treatment
While there is currently no cure for diabetes, it is highly manageable. The primary goal is to keep blood sugar levels as close to a target range as possible to prevent complications.
1.	Lifestyle Changes:
o	Diet: Eating a balanced, heart-healthy diet, with a focus on controlling carbohydrate intake (especially simple sugars).
o	Exercise: Getting regular physical activity helps the body's cells use glucose more efficiently.
o	Weight Management: Achieving and maintaining a healthy weight is crucial, especially for Type 2 diabetes.
2.	Blood Sugar Monitoring:
o	Regularly checking blood glucose levels with a home monitor (glucometer) or a Continuous Glucose Monitor (CGM).
o	Getting a periodic A1C test (measures average blood sugar over the past 2-3 months).
3.	Medication:
o	Insulin: Essential for all Type 1 patients and many Type 2 patients.
o	Oral Medications: Used for Type 2 diabetes to help the body use insulin better (e.g., Metformin) or to stimulate the pancreas to produce more insulin.

APPENDIX:
The term "appendix" usually refers to the vermiform appendix, a small, tube-shaped pouch attached to the large intestine in the lower right abdomen.
  The Appendix Organ
•	Location: It is located near the junction of the small and large intestines (the cecum), in the lower right part of the abdomen.
•	Function (Debated): For a long time, the appendix was considered a vestigial organ (an evolutionary remnant with no purpose). However, recent research suggests it may play a role in the human body:
o	Immune System: It contains lymphoid tissue, suggesting it has a function in the immune system.
o	"Safe House" for Gut Flora: A popular theory suggests it acts as a reservoir for beneficial gut bacteria, helping to repopulate the digestive system after a severe diarrheal illness has flushed out the normal flora.
•	Removal: Despite its possible functions, removing the appendix (appendectomy) generally causes no noticeable ill effects on human health.
   Diagnosis
Diagnosis is based on a physical exam, a history of symptoms, and may be confirmed with imaging tests like:
•	Ultrasound
•	CT scan (Computed Tomography)
•	MRI (Magnetic Resonance Imaging)
•	Blood and urine tests are also common.
Treatment
The standard treatment for appendicitis is surgical removal of the appendix, called an appendectomy.
•	Surgery: This is typically performed using minimally invasive laparoscopic surgery (using small incisions and a camera) or, less commonly, an open incision (laparotomy).
•	Antibiotics: Antibiotics are given to treat the infection. In some early, uncomplicated cases, doctors may treat with antibiotics alone, though surgery remains the standard of care to prevent recurrence and rupture.
•	Burst Appendix: If the appendix has ruptured, immediate surgery is required to remove the appendix and clean the abdominal cavity to prevent or treat peritonitis (a severe infection of the abdominal lining
KIDNEY STONE:
Calcium Stones (Most Common) 
High levels of calcium or oxalate in the urine. Risk factors include: not drinking enough water, high-sodium or high-protein diets, and certain metabolic conditions.
Uric Acid Stones
High levels of acid in the urine, often linked to a high-protein diet (high purine intake), gout, and not drinking enough water.Struvite StonesResult from urinary tract infections (UTIs). These stones can grow quickly and become quite large.
Cystine Stones
Caused by a rare hereditary disorder (cystinuria) that causes an amino acid (cystine) to leak into the urine.
Symptoms
Small stones may pass without noticeable symptoms. However, if a stone moves into the narrow tubes (ureters) connecting the kidney and bladder, it can cause severe pain.
Symptom	Description
Severe Pain (Renal Colic)	Pain that comes in waves and is often excruciating. It typically begins in the side and back, below the ribs, and radiates to the lower abdomen and groin.
Painful Urination	Pain or a burning sensation when urinating.
Nausea and Vomiting	Common due to the shared nerve pathways between the kidneys and stomach.
Blood in Urine	Urine may appear pink, red, or brown (hematuria).
Other Symptoms	Frequent urge to urinate, passing small amounts of urine, or fever and chills (if an infection is present—a medical emergency).
Treatment Options
Treatment depends on the stone's size, type, location, and whether it's causing a blockage or infection.
For Small Stones (May Pass Naturally)
•	Hydration: Drinking lots of water (up to 2-3 quarts a day) helps flush the stone out.
•	Pain Relievers: Over-the-counter or prescription medications for pain and discomfort.
•	Alpha Blockers: Medications (like tamsulosin) that relax the muscles in the ureter to help the stone pass more quickly and easily.
For Large Stones (Require Intervention)
If a stone is too large to pass, is blocking the urinary tract, or is causing uncontrolled pain or infection, a urologist will perform one of the following procedures:
1.	Extracorporeal Shock Wave Lithotripsy (ESWL):
o	Method: Uses focused sound waves aimed from outside the body to break the stone into tiny pieces, which can then be passed in the urine.
o	Best for: Smaller stones in the kidney or upper ureter.
2.	Ureteroscopy (URS):
o	Method: A thin, lighted tube (ureteroscope) is passed through the urethra and bladder up to the ureter. The stone is either removed directly with a tiny basket or broken up with a laser.
o	Best for: Stones stuck in the ureter or those not suitable for ESWL. A temporary stent may be placed to keep the ureter open.
3.	Percutaneous Nephrolithotomy (PCNL):
o	Method: Used for very large or complex stones. The surgeon makes a small incision in the back, inserts a tube, and uses specialized instruments to directly remove the stone or break it up.
Prevention
To reduce the risk of recurrence:
•	Increase Fluid Intake: Drink enough water (ideally 2-3 liters/day) to produce clear or very light-colored urine.
•	Dietary Changes: Specific changes depend on the stone type, but common recommendations include:
o	Limiting sodium (salt) intake.
o	Reducing high-purine foods (for uric acid stones).
o	Maintaining adequate, but not excessive, calcium intake.
•	Medication: Depending on the stone type, your doctor may prescribe medications (like potassium citrate or allopurinol) to help regulate the concentration of stone-forming substances in your urine.
EYEPAIN:
Eye pain, medically known as ophthalmalgia, can range from mild irritation to a sign of a serious medical condition. It can be categorized as pain on the surface of the eye (ocular pain) or pain within the eye (orbital pain).
It is important to seek immediate medical attention if you experience severe eye pain or pain accompanied by sudden vision loss.

The cause of the pain often depends on whether it's surface pain or internal pain.
Surface (Ocular) Pain
This type of pain is usually described as burning, itching, or a feeling of a foreign body in the eye.
Common Causes	Description
Foreign Body	A speck of dust, sand, or an eyelash in the eye.
Conjunctivitis (Pink Eye)	Inflammation of the clear layer (conjunctiva) covering the white part of the eye. Causes redness, discharge, and foreign body sensation.
Corneal Abrasion	A scratch on the cornea (the clear front surface of the eye), often caused by trauma, a foreign body, or improper contact lens use.
Dry Eyes	The eyes don't produce enough lubrication, leading to a gritty, burning, or stinging sensation.
Blepharitis / Stye	Inflammation or infection of the eyelid margins or oil glands, causing a painful, red lump.
Contact Lens Issues	Wearing lenses too long, not cleaning them properly, or using the wrong solution.

Internal (Orbital) Pain
This pain is usually deeper, throbbing, or aching, and may be more serious.
Serious Causes	Description
Acute Angle-Closure Glaucoma	A sudden, rapid increase in the pressure inside the eye, which is a medical emergency that can cause vision loss. Symptoms include severe pain, headache, nausea/vomiting, and seeing halos around lights.
Iritis/Uveitis	Inflammation of the iris or the inner layers of the eye (uvea), often causing pain, light sensitivity, and blurry vision.
Optic Neuritis	Inflammation of the optic nerve, which connects the back of the eye to the brain. Often causes pain when moving the eye and vision loss.
Sinusitis/Migraine	Infections in the sinuses or a severe migraine can cause a headache and pain behind or around the eye, which feels like it's originating internally.



When to Seek Medical Attention
While many minor causes of eye pain can be treated at home, certain symptoms require immediate evaluation by an eye doctor (optometrist or ophthalmologist) or a visit to the emergency room.
Seek Emergency Medical Care Immediately If You Have:	See a Doctor Soon (Within 24 Hours) If You Have:
Severe, sudden pain	Pain that is persistent or does not improve within a day or two.
Sudden loss or change in vision	Pain accompanied by eye redness or discharge.
Pain with nausea and vomiting (especially with blurry vision)	Pain due to a contact lens problem.
Halos or colored rings around lights	Pain with mild light sensitivity.
Chemical splash or a high-velocity foreign object/injury (e.g., metal shard)	Mild pain accompanied by an eyelid bump (stye).
Trouble moving the eye or keeping it open	You have had recent eye surgery or an eye injection.
Fever or chills along with eye pain and redness	
General Treatments and Home Care
Treatment is always based on the underlying cause, but here are general recommendations:
Treatment/Action	What It Helps Treat
Artificial Tears (over-the-counter eye drops)	Dry eyes, eye strain, minor irritation.
Rest (Avoiding screens/reading)	Eye strain, general fatigue.
Warm Compress	Styes or blepharitis (helps unclog oil glands).
Cold Compress	Conjunctivitis, allergies, or minor swelling.
Stop Wearing Contacts	Give the cornea time to heal; use glasses until symptoms resolve.
Flushing Eye with Water	Chemical exposure (for 15-20 minutes) or minor foreign bodies. Do not rub your eye.
Prescription Medications	Antibiotic/Antiviral Drops: For infections like bacterial conjunctivitis. Steroid/NSAID Drops: To reduce inflammation from uveitis or post-injury. Glaucoma Medication: To lower internal eye pressure.

GET BITTEN BY DOG AND SNAKE:
     Getting bitten by an animal or a snake is a serious and potentially life-threatening emergency. You must seek immediate medical attention by calling your local emergency services or going to the emergency room right away.
Here is the essential first aid guidance for both a dog bite and a snake bite on a human:
1. Snake Bite: IMMEDIATE EMERGENCY
A snake bite must be treated as a medical emergency, even if you are unsure if the snake was venomous. Do NOT wait for symptoms to appear.
DO	DO NOT
Call Emergency Services immediately.	DO NOT apply a tourniquet or any form of constriction bandage. This can cause tissue damage.
Stay calm and still. Minimal movement helps slow the spread of venom.	DO NOT cut the wound or attempt to suck out the venom. This is ineffective and can cause infection.
Remove any jewelry, watches, or tight clothing near the bite area before swelling begins.	DO NOT apply ice or immerse the wound in water. It can worsen the tissue damage.
Sit or lie down so the bite is in a neutral or slightly below the heart position, if possible.	DO NOT drink alcohol or caffeine.
Gently wash the bite area with soap and water and cover it with a clean, dry bandage.	DO NOT take pain relievers like aspirin or ibuprofen, as they can increase the risk of bleeding.
Try to remember the snake's color and shape for identification, but DO NOT try to catch or kill it.	


Dog Bite: URGENT MEDICAL CARE
A dog bite carries a risk of infection (including bacterial infection and rabies) and tissue damage. Seek prompt medical care for any bite that breaks the skin.
DO	DO NOT
Stop the bleeding. Apply gentle pressure with a clean cloth or sterile bandage.	DO NOT delay medical care for a deep or severe wound.
Wash the wound thoroughly with soap and water for 5 to 10 minutes.	DO NOT underestimate the risk of infection, especially with puncture wounds.
Apply an antibiotic cream or ointment and cover the bite with a sterile bandage.	DO NOT assume the dog is vaccinated; try to get the owner's information for rabies status.
Seek medical attention if the skin is broken, the wound is deep, bleeding is severe, or if the dog is unknown/unvaccinated against rabies.	
	

Key Medical Concerns for Dog Bites:
•	Infection: Dog bites frequently become infected and often require prescription antibiotics.
•	Rabies: Rabies is a fatal viral disease. You will need to confirm the dog's vaccination status. If rabies is a possibility, a post-exposure vaccination series may be necessary.
•	Tetanus: A tetanus booster may be needed if your last shot was more than five years ago.


Get stepped on a nail:
Stepping on a rusted nail is a serious injury that carries a high risk of infection, most notably tetanus.
You should seek prompt medical attention at an urgent care clinic or your doctor's office, especially to address your tetanus vaccination status.


Immediate First Aid Steps
1.	Remove the nail (if it's not deep): If the nail is still in your foot and it's easy to remove, pull it out gently. If the nail is deeply embedded or causes severe bleeding, leave it in place and seek emergency medical care immediately.
2.	Stop the bleeding: Apply gentle pressure with a clean cloth or sterile bandage until any bleeding stops. Elevating your foot slightly may help.
3.	Clean the wound thoroughly:
o	Wash your hands first.
o	Rinse the wound with clean, running water and mild soap for at least 5 to 10 minutes to flush out dirt, rust, and bacteria.
o	If dirt or debris remains, use clean tweezers or a washcloth to gently scrub it off.
4.	Apply Antibiotic Ointment: Put a thin layer of an over-the-counter antibiotic cream or ointment on the wound.
5.	Cover the wound: Apply a clean bandage or sterile gauze to protect the area

Crucial Medical Concerns
A puncture wound from a dirty or rusty object, like a nail, is considered a high-risk injury due to the chance of infection.
1. Tetanus Risk
The bacteria that causes tetanus (sometimes called lockjaw) lives in soil, dust, and animal feces, and a rusty nail is a common way for it to enter a deep wound.
•	You need to confirm when you had your last tetanus booster shot.
•	See a healthcare provider if:
o	It has been more than 5 years since your last tetanus shot and the wound is dirty (like from a rusty nail).
o	It has been more than 10 years since your last tetanus shot.
o	You don't know your vaccination history.
•	The tetanus booster is most effective if given within 48 hours of the injury.
2. General Infection
Puncture wounds are deep and narrow, making them difficult to clean and creating an environment where bacteria can thrive. You may need X-rays to check for debris (like pieces of sock or the nail tip) and antibiotics to prevent a serious infection.




When to Seek Medical Care Immediately (Urgent Care or ER)
Go to a healthcare provider within 24 hours for a thorough cleaning and to assess your tetanus status.
Go to the Emergency Room (ER) or call emergency services if:
•	The nail is deeply embedded and you cannot remove it, or it is causing severe bleeding.
•	You experience severe pain or swelling that rapidly worsens.
•	You develop signs of infection (increasing redness, swelling, warmth, pus, fever, or red streaks extending from the wound).
•	You lose feeling in your foot.
•	You are an individual with a compromised immune system (e.g., have diabetes)

MEDICAL SCHEMES:
 Chief Minister's Comprehensive Health Insurance Scheme (CMCHIS)
This is the most comprehensive scheme for the general public and is the primary avenue for accessing free treatment for diseases.
________________________________________
1. Chief Minister's Comprehensive Health Insurance Scheme (CMCHIS)
•	Objective: To provide free, quality healthcare to the economically weaker sections of the society through a cashless process at empanelled hospitals.
•	Benefits:
o	Coverage: Up to ₹5,00,000 (Five Lakhs) per family per year on a floater basis.
o	Procedures Covered: Covers a wide range of treatments, including over 1,090 procedures, 8 follow-up treatments, and 52 diagnostic procedures.
o	Specific Diseases/Treatments Covered: It includes expensive and critical treatments like:
	Cancer Treatment (Oncology)
	Cardiac (Heart) Surgeries and Procedures
	Neurological Disorders
	Renal (Kidney) Failure Treatments (like Dialysis)
	Organ Transplantations (with specific sub-limits)
	Maternity and Neonatal Care
	General Surgeries and Medical Therapy
o	Cashless Treatment: Treatment is cashless at a large network of empanelled Government and Private Hospitals across Tamil Nadu.
o	Pre-existing conditions are covered from day one.
•	Eligibility Criteria (Key Points):
o	Must be a resident of Tamil Nadu.
o	The family's annual income should not exceed ₹1,20,000 (One Lakh Twenty Thousand) per annum.
o	Your name must be included in the family's Ration Card.
o	Members of Unorganised Labour Welfare Boards are also eligible.
•	Other Key Health Initiatives Focused on Disease Management
•	The Tamil Nadu government also runs specific programs that focus on management and prevention of certain diseases:
Scheme Name	Focus / Benefit	Target Diseases / Groups
Makkalai Thedi Maruthuvam (Medicine at People's Doorstep)	Provides doorstep healthcare services, including screening, treatment, and free delivery of medicines.	Primarily for Non-Communicable Diseases (NCDs) like Diabetes, Hypertension (High Blood Pressure), and mobility-impaired individuals (e.g., for physiotherapy).
Innuyir Kappom Thittam	Provides free, immediate medical treatment for road accident victims within the first 48 hours to save lives and prevent long-term disability.	Accident & Trauma cases (treatment covered up to ₹2,00,000).
Specialized Clinics/Funds	Provides financial aid and specialised treatment.	Cancer: Financial support schemes like the Health Minister's Cancer Patients Fund (for BPL patients at enlisted Regional Cancer Centres).















"""
]

# --- OPTION 1: REBUILD THE DATABASE FROM SCRATCH ---
# This is the simplest method. It's best for smaller datasets or when you want to ensure
# all data is processed consistently.
def rebuild_database_from_scratch(existing_data, new_data):
    print("--- Rebuilding vector database from scratch ---")
    
    # Combine old and new data
    combined_data = existing_data + new_data
    
    # Split the combined data into documents
    documents = text_splitter.create_documents(combined_data)
    
    # Create the new vector store
    print("Creating new vector store from combined data...")
    new_vector_store = FAISS.from_documents(documents, embeddings)
    
    # Save the new vector store, overwriting the old one
    new_vector_store.save_local(db_path)
    print(f"Database rebuilt and saved to '{os.path.abspath(db_path)}'")


# --- OPTION 2: ADD TO AN EXISTING DATABASE ---
# This method is more efficient for large datasets, as it avoids re-processing old data.
def add_to_existing_database(new_data):
    print("\n--- Adding data to existing vector database ---")
    
    # Check if the database file exists
    if not os.path.exists(db_path):
        print("Error: Existing database not found. Please run db_create.py first.")
        return

    # Load the existing vector database
    print("Loading existing vector database...")
    existing_vector_store = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)

    # Convert the new data into documents
    new_documents = text_splitter.create_documents(new_data)
    
    # Add the new documents to the existing vector store
    print("Adding new documents...")
    existing_vector_store.add_documents(new_documents)
    
    # Save the updated database
    existing_vector_store.save_local(db_path)
    print(f"New data added and database updated at '{os.path.abspath(db_path)}'")


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # --- IMPORTANT: CHOOSE ONE OPTION TO RUN AT A TIME ---
    # For this demonstration, we'll run the "add to existing" method.
    # Make sure you've already created the initial database using `db_create.py`.

    # Example medical data (same as from the original db_create.py)
    existing_medical_data = [
        "Aspirin, a nonsteroidal anti-inflammatory drug (NSAID), is used to treat pain, fever, and inflammation. It can also be used as an antiplatelet agent to prevent blood clots.",
        "The pancreas is an organ located in the abdomen. It plays a crucial role in converting the food we eat into fuel for the body's cells. It has two main functions: an exocrine function that helps in digestion and an endocrine function that regulates blood sugar.",
        "Hypertension, or high blood pressure, is a condition where the force of the blood against the artery walls is too high. It can lead to serious health problems, including heart disease and stroke. It can be managed with lifestyle changes and medication.",
        "Diabetes mellitus is a chronic condition that affects how your body turns food into energy. It is characterized by high blood glucose (sugar) levels. There are two main types: Type 1 and Type 2."
    ]

    # Run the function to add data to the existing database
    add_to_existing_database(new_medical_data)

    # To use the rebuild method, you would uncomment the line below.
    # rebuild_database_from_scratch(existing_medical_data, new_medical_data)
