"""Small dependency-free retrieval knowledge base for PlantVision AI.

This is a starter RAG corpus, not an authoritative pesticide guide. Expand it with
reviewed agricultural-extension documents before relying on it for field decisions.
"""
import re
import math

KNOWLEDGE_DOCUMENTS = [
    {
        "id": "PV-GENERAL-01",
        "title": "General plant disease prevention",
        "source": "University of Minnesota Extension",
        "source_title": "Preventing plant diseases in the garden",
        "source_url": "https://extension.umn.edu/garden-and-home/yard-and-garden/gardening-in-minnesota/yard-and-garden-problems/preventing-plant-diseases-in-the-garden",
        "source_publisher": "University of Minnesota Extension",
        "text": (
            "General prevention: inspect plants regularly; avoid moving from infected "
            "plants to healthy plants without cleaning hands and tools; remove diseased "
            "plant material where appropriate and dispose of it according to local guidance; "
            "use clean planting material; improve spacing and airflow where suitable; water "
            "the root zone when practical and avoid prolonged leaf wetness; clean containers "
            "and tools; rotate crops only when the disease organism and crop rotation make "
            "that useful. These are general practices, not a diagnosis-specific prescription."
        ),
    },
    {
        "id": "PV-FUNGAL-01",
        "title": "General fungal leaf disease management",
        "source": "University of Minnesota Extension",
        "source_title": "Preventing plant diseases in the garden",
        "source_url": "https://extension.umn.edu/garden-and-home/yard-and-garden/gardening-in-minnesota/yard-and-garden-problems/preventing-plant-diseases-in-the-garden",
        "source_publisher": "University of Minnesota Extension",
        "text": (
            "For suspected fungal leaf diseases, check whether spots expand, whether lesions "
            "have distinctive margins or growth, and whether symptoms appear on new leaves. "
            "Avoid overhead watering when practical, allow foliage to dry, maintain suitable "
            "airflow, sanitize tools, and manage infected debris where recommended for that "
            "crop and disease. Several disorders look alike, so confirm uncertain cases. "
            "Do not apply fungicides without checking local registered guidance and labels."
        ),
    },
    {
        "id": "PV-BACTERIAL-01",
        "title": "General bacterial plant disease precautions",
        "source": "University of Minnesota Extension",
        "source_title": "Preventing plant diseases in the garden",
        "source_url": "https://extension.umn.edu/garden-and-home/yard-and-garden/gardening-in-minnesota/yard-and-garden-problems/preventing-plant-diseases-in-the-garden",
        "source_publisher": "University of Minnesota Extension",
        "text": (
            "For suspected bacterial plant disease, avoid handling wet plants, clean tools "
            "between plants, use clean planting material, reduce splashing water where "
            "practical, and remove affected material only when recommended for the crop. "
            "Bacterial and fungal spots can look similar. Seek local expert confirmation "
            "before choosing a treatment; some products do not cure infected plant tissue."
        ),
    },
    {
        "id": "PV-VIRAL-01",
        "title": "General viral plant disease precautions",
        "source": "University of Minnesota Extension",
        "source_title": "Preventing plant diseases in the garden",
        "source_url": "https://extension.umn.edu/garden-and-home/yard-and-garden/gardening-in-minnesota/yard-and-garden-problems/preventing-plant-diseases-in-the-garden",
        "source_publisher": "University of Minnesota Extension",
        "text": (
            "For suspected viral disease, inspect for mosaic patterns, curling, distortion, "
            "or stunting, but these signs are not definitive. Control insect vectors only "
            "with locally appropriate integrated pest management advice, sanitize tools "
            "when relevant, use clean planting material, and ask an agricultural expert "
            "about removal of infected plants. There is generally no curative treatment "
            "for a plant once systemically infected by a virus."
        ),
    },
    {
        "id": "PV-POTATO-EARLY-01",
        "title": "Potato early blight: cautious field checks",
        "source": "University of Minnesota Extension",
        "source_title": "Potato leaf spots: early blight and late blight",
        "source_url": "https://apps.extension.umn.edu/garden/diagnose/plant/vegetable/potato/leavesspots.html",
        "source_publisher": "University of Minnesota Extension",
        "text": (
            "Potato early blight is commonly associated with brown leaf lesions that may "
            "show concentric rings, often on older leaves. Similar symptoms can have other "
            "causes. Reduce plant stress with appropriate irrigation and nutrition, avoid "
            "prolonged leaf wetness where practical, maintain suitable spacing, remove crop "
            "residues according to local recommendations, and rotate crops where suitable. "
            "Use locally approved disease-management recommendations if disease is confirmed."
        ),
    },
    {
        "id": "PV-POTATO-LATE-01",
        "title": "Potato late blight: urgent escalation",
        "source": "University of Minnesota Extension",
        "source_title": "Potato leaf spots: early blight and late blight",
        "source_url": "https://apps.extension.umn.edu/garden/diagnose/plant/vegetable/potato/leavesspots.html",
        "source_publisher": "University of Minnesota Extension",
        "text": (
            "Potato late blight can progress rapidly in favorable cool, wet conditions and "
            "may cause water-soaked lesions that turn brown or dark, sometimes with pale "
            "growth at lesion margins in humid conditions. This is not a visual diagnosis. "
            "If rapid spread or suspected late blight occurs, isolate work in affected areas "
            "where practical and contact local agricultural extension promptly. Avoid moving "
            "infected plant material. Follow locally registered guidance rather than "
            "improvised pesticide treatments."
        ),
    },
    {
        "id": "PV-APPLE-SCAB-01",
        "title": "Apple scab: cautious field checks",
        "source": "University of Minnesota Extension",
        "source_title": "Apple scab of apples and crabapples",
        "source_url": "https://extension.umn.edu/garden-and-home/yard-and-garden/gardening-in-minnesota/yard-and-garden-problems/apple-scab",
        "source_publisher": "University of Minnesota Extension",
        "text": (
            "Apple scab can cause olive-green to dark scabby lesions on leaves and fruit; "
            "leaf symptoms and fruit marks vary with stage and conditions. Similar marks "
            "can have other causes. Monitor leaves and fruit, remove fallen infected leaves "
            "where recommended, improve airflow with appropriate pruning, and follow local "
            "orchard sanitation and disease-management guidance. Do not treat a model output "
            "as a confirmed diagnosis."
        ),
    },
    {
        "id": "PV-CORN-RUST-01",
        "title": "Corn common rust: cautious field checks",
        "source": "Penn State Extension",
        "source_title": "Not every spot is the same: identifying corn leaf diseases",
        "source_url": "https://extension.psu.edu/not-every-spot-is-the-same-identifying-corn-leaf-diseases",
        "source_publisher": "Penn State Extension",
        "text": (
            "Corn common rust may produce raised rust-colored pustules on leaf surfaces. "
            "Other leaf diseases can resemble rust. Monitor spread across leaves, record "
            "conditions, and consult local crop advisers about resistant hybrids and "
            "region-specific management. Avoid recommending a chemical treatment solely "
            "from an image classifier."
        ),
    },
    {
        "id": "PV-TOMATO-LEAF-01",
        "title": "Tomato leaf spot and blight: general checks",
        "source": "UC Statewide IPM Program",
        "source_title": "Tomato pest management guidelines",
        "source_url": "https://ipm.ucanr.edu/agriculture/tomato/",
        "source_publisher": "UC Statewide IPM Program",
        "text": (
            "Tomato leaf spots and blights have overlapping visual symptoms. Check the "
            "location and pattern of lesions, lower versus upper leaves, stems and fruit, "
            "and whether symptoms are spreading. Water near the soil when practical, "
            "maintain airflow, sanitize tools, avoid working among wet plants, and manage "
            "plant debris according to local recommendations. Confirm the likely cause "
            "before selecting disease-specific treatment."
        ),
    },

    {
        "id": "PV-APPLE-BLACK-ROT-01",
        "title": "Apple black rot: symptoms and field checks",
        "source": "University of Minnesota Extension",
        "source_title": "Apple leaves: spots and blotches",
        "source_url": "https://apps.extension.umn.edu/garden/diagnose/plant/fruit/apple/leavesspots.html",
        "source_publisher": "University of Minnesota Extension",
        "text": (
            "Apple black rot may cause round leaf spots with purple borders and tan centers. "
            "Fruit may develop larger brown spots with rings, remain firm while rotting, or "
            "become mummified on the tree. Branch cankers may also occur. These signs can "
            "overlap with other problems; confirm the diagnosis before choosing treatment. "
            "Consult local extension guidance for sanitation and pruning recommendations."
        ),
    },
    {
        "id": "PV-APPLE-CEDAR-RUST-01",
        "title": "Apple cedar rust: symptoms and field checks",
        "source": "University of Minnesota Extension",
        "source_title": "Apple leaves: spots and blotches",
        "source_url": "https://apps.extension.umn.edu/garden/diagnose/plant/fruit/apple/leavesspots.html",
        "source_publisher": "University of Minnesota Extension",
        "text": (
            "Cedar-apple rust can cause yellow spots that become bright orange and develop "
            "a red border. Small dark structures may appear on leaf surfaces, and fruit can "
            "develop rough green-to-brown marks. Similar symptoms have other causes; use "
            "the plant and nearby host context and seek local expert confirmation before "
            "taking disease-specific action."
        ),
    },
    {
        "id": "PV-GRAPE-BLACK-ROT-01",
        "title": "Grape black rot: symptoms and sanitation",
        "source": "Penn State Extension",
        "source_title": "Grape disease: black rot",
        "source_url": "https://extension.psu.edu/grape-disease-black-rot",
        "source_publisher": "Penn State Extension",
        "text": (
            "Grape black rot may cause leaf spots and fruit that shrivels into hard, black "
            "mummies. The fungus can survive in mummified fruit and cane lesions, and wet "
            "conditions can support infection. Extension guidance recommends removing "
            "infected prunings and mummified berries before new growth where appropriate. "
            "Local climate, cultivar, and production system affect management decisions."
        ),
    },
    {
        "id": "PV-CITRUS-HLB-01",
        "title": "Citrus greening (HLB): warning signs",
        "source": "UC Statewide IPM Program",
        "source_title": "Asian citrus psyllid and huanglongbing disease",
        "source_url": "https://ipm.ucanr.edu/home-and-landscape/asian-citrus-psyllid-and-huanglongbing-disease/",
        "source_publisher": "UC Statewide IPM Program",
        "text": (
            "Huanglongbing, also called citrus greening, can cause uneven blotchy yellow "
            "mottling on leaves, small or lopsided fruit, premature fruit drop, and twig "
            "dieback. Nutrient deficiencies and other problems can also yellow leaves. "
            "A leaf-image prediction cannot confirm HLB; contact a local plant-health "
            "authority or agricultural extension service for appropriate testing and advice."
        ),
    },
    {
        "id": "PV-PEPPER-BACTERIAL-SPOT-01",
        "title": "Pepper bacterial spot: symptoms and prevention",
        "source": "UC Statewide IPM Program",
        "source_title": "Bacterial spot on peppers",
        "source_url": "https://ipm.ucanr.edu/agriculture/peppers/bacterial-spot/",
        "source_publisher": "UC Statewide IPM Program",
        "text": (
            "Pepper bacterial spot may cause small, initially water-soaked angular leaf "
            "spots that later turn brown, raised stem lesions, and rough raised fruit spots. "
            "The pathogen can be seedborne and spread on transplants; moisture favors "
            "disease development. Use clean, pathogen-tested seed or healthy transplants, "
            "avoid prolonged leaf wetness where practical, and follow locally applicable "
            "extension guidance. Do not select a chemical treatment from an image alone."
        ),
    },
    {
        "id": "PV-TOMATO-BACTERIAL-SPOT-01",
        "title": "Tomato bacterial spot: symptoms and prevention",
        "source": "UC Statewide IPM Program",
        "source_title": "Bacterial spot on tomato",
        "source_url": "https://ipm.ucanr.edu/agriculture/tomato/bacterial-spot/",
        "source_publisher": "UC Statewide IPM Program",
        "text": (
            "Tomato bacterial spot may cause water-soaked areas on older leaves that turn "
            "dark brown or black. Fruit can develop slightly sunken spots that later become "
            "brown and scabby. The pathogen can be seedborne and survive in crop debris or "
            "alternate hosts. Use clean seed and transplants, reduce splashing water where "
            "practical, and confirm the diagnosis before choosing treatment."
        ),
    },
    {
        "id": "PV-TOMATO-EARLY-BLIGHT-01",
        "title": "Tomato early blight: symptoms and prevention",
        "source": "UC Statewide IPM Program",
        "source_title": "Early blight on tomatoes",
        "source_url": "https://ipm.ucanr.edu/home-and-landscape/early-blight-on-tomatoes/",
        "source_publisher": "UC Statewide IPM Program",
        "text": (
            "Tomato early blight may cause brown or black spots with concentric rings, "
            "often beginning on older leaves; stems and fruit can also be affected. Leaf "
            "wetness favors infection. Avoid overhead irrigation where practical, maintain "
            "airflow, and rotate crops when suitable for the growing system. Confirm the "
            "cause of spots before disease-specific treatment."
        ),
    },
    {
        "id": "PV-TOMATO-LATE-BLIGHT-01",
        "title": "Tomato late blight: rapid-spread warning signs",
        "source": "UC Statewide IPM Program",
        "source_title": "Late blight on tomato",
        "source_url": "https://ipm.ucanr.edu/agriculture/tomato/late-blight/",
        "source_publisher": "UC Statewide IPM Program",
        "text": (
            "Tomato late blight may cause water-soaked leaf areas that rapidly enlarge into "
            "dark purple-brown blotches. Pale growth may appear near lesions in humid "
            "conditions, and stems or fruit may be affected. The disease can spread rapidly "
            "under favorable conditions. If rapid spread is suspected, limit movement of "
            "plant material and contact local agricultural extension promptly."
        ),
    },
    {
        "id": "PV-TOMATO-TYLCV-01",
        "title": "Tomato yellow leaf curl virus: warning signs",
        "source": "UC Statewide IPM Program",
        "source_title": "Tomato yellow leaf curl",
        "source_url": "https://ipm.ucanr.edu/agriculture/tomato/tomato-yellow-leaf-curl/",
        "source_publisher": "UC Statewide IPM Program",
        "text": (
            "Tomato yellow leaf curl virus may cause upward-curling, crumpled leaves with "
            "yellowing between or along veins, shortened internodes, and stunted upright "
            "growth. Whiteflies can transmit the virus. Similar symptoms can have other "
            "causes; do not treat a classifier output as confirmation. Seek local expert "
            "advice about testing, resistant varieties, and vector management."
        ),
    },
    {
        "id": "PV-TOMATO-MOSAIC-01",
        "title": "Tomato mosaic virus: symptoms and caution",
        "source": "UC Statewide IPM Program",
        "source_title": "Plant diseases: virus symptoms you may see on tomato",
        "source_url": "https://ipm.ucanr.edu/agriculture/tomato/diseases-virus-symptoms-you-may-see/",
        "source_publisher": "UC Statewide IPM Program",
        "text": (
            "Tomato mosaic virus symptoms vary by cultivar and virus strain. Mosaic "
            "coloring and leaf malformation can occur, but visual symptoms overlap with "
            "other viral, pest, and nutrient problems. Use clean planting material, clean "
            "tools where relevant, and seek local diagnostic advice before removing plants "
            "or selecting a treatment."
        ),
    },
    {
        "id": "PV-SPIDER-MITES-01",
        "title": "Tomato spider mites: signs and nonchemical checks",
        "source": "UC Statewide IPM Program",
        "source_title": "Spider mites",
        "source_url": "https://ipm.ucanr.edu/home-and-landscape/spider-mites/",
        "source_publisher": "UC Statewide IPM Program",
        "text": (
            "Spider mites may cause fine pale stippling, bronzing, yellowing, leaf drop, "
            "and sometimes webbing. Check leaf undersides with a hand lens because mites "
            "are tiny. Water-stressed plants and hot, dusty conditions can worsen damage. "
            "Confirm mites are present before treating and protect beneficial predators; "
            "follow local guidance for any control product."
        ),
    },
    {
        "id": "PV-SQUASH-POWDERY-01",
        "title": "Squash powdery mildew: symptoms and prevention",
        "source": "University of Minnesota Extension",
        "source_title": "Powdery mildew of cucurbits",
        "source_url": "https://extension.umn.edu/agriculture/specialty-crops/vegetable-farming/disease-management/powdery-mildew-of-cucurbits",
        "source_publisher": "University of Minnesota Extension",
        "text": (
            "Cucurbit powdery mildew can begin as pale yellow spots and develop white-to-gray "
            "powdery patches on leaf surfaces. Severe infection weakens vines and can lead "
            "to premature fruit ripening. Use suitable spacing, improve air movement, "
            "control weeds, and consider resistant varieties. Confirm the disease before "
            "using a disease-specific treatment."
        ),
    },

    {
        "id": "PV-CORN-LEAF-DISEASES-01",
        "title": "Corn leaf diseases: gray leaf spot and northern leaf blight",
        "source": "Penn State Extension",
        "source_title": "Not every spot is the same: identifying corn leaf diseases",
        "source_url": "https://extension.psu.edu/not-every-spot-is-the-same-identifying-corn-leaf-diseases",
        "source_publisher": "Penn State Extension",
        "text": (
            "Corn gray leaf spot can begin as tan lesions with yellow halos and develop into "
            "longer rectangular gray-to-tan lesions aligned with leaf veins. Corn leaf "
            "diseases can resemble one another, and symptom appearance varies by hybrid and "
            "conditions. Compare the full symptom pattern and seek local crop-adviser "
            "confirmation before selecting disease-specific treatment, including for "
            "suspected northern leaf blight."
        ),
    },

    {
        "id": "PV-CHERRY-POWDERY-01",
        "title": "Cherry powdery mildew: symptoms and prevention",
        "source": "Pacific Northwest Plant Disease Management Handbook",
        "source_title": "Cherry (Prunus spp.) — powdery mildew",
        "source_url": "https://pnwhandbooks.org/plantdisease/host-disease/cherry-prunus-spp-powdery-mildew",
        "source_publisher": "Pacific Northwest Plant Disease Management Handbook",
        "text": (
            "Cherry powdery mildew may begin as light-green circular lesions on young leaves, "
            "with subtle white cotton-like growth. Severe infection can curl or blister leaves, "
            "and fruit may develop depressed areas or white growth. Dense foliage, warm "
            "temperatures, and high humidity can favor disease. Promote air circulation, avoid "
            "wetting leaves where practical, and seek local orchard guidance before treatment."
        ),
    },
    {
        "id": "PV-GRAPE-ESCA-01",
        "title": "Grape Esca (Black Measles): signs and cautions",
        "source": "UC Statewide IPM Program",
        "source_title": "Esca (Black Measles) on grape",
        "source_url": "https://ipm.ucanr.edu/agriculture/grape/esca-black-measles/",
        "source_publisher": "UC Statewide IPM Program",
        "text": (
            "Esca, also called black measles, is a grapevine trunk disease. Leaf symptoms can "
            "include interveinal striping that becomes dry and necrotic; berries may develop "
            "small dark spots with brown-purple rings. Symptoms vary between years and can "
            "resemble other grapevine problems. Suspected trunk disease needs vineyard-level "
            "assessment; fungicide cannot eradicate an established wood infection."
        ),
    },
    {
        "id": "PV-GRAPE-LEAF-BLIGHT-01",
        "title": "Grape leaf blight (Isariopsis leaf spot): field signs",
        "source": "NC State Extension",
        "source_title": "North Carolina Winegrape Growers Guide — Pest Management",
        "source_url": "https://content.ces.ncsu.edu/north-carolina-winegrape-growers-guide/chapter-8-pest-management",
        "source_publisher": "North Carolina State University Extension",
        "text": (
            "Grape leaf blight, also called Isariopsis leaf spot, is associated with "
            "Pseudocercospora vitis. Lesions may begin as irregular dull-red or brown spots "
            "and become black and brittle as they age, often later in the season. Other grape "
            "leaf spots can look similar. Promote air circulation and consult local grape "
            "extension guidance for diagnosis and region-appropriate management."
        ),
    },
    {
        "id": "PV-PEACH-BACTERIAL-SPOT-01",
        "title": "Peach bacterial spot: symptoms and caution",
        "source": "Penn State Extension",
        "source_title": "2026 disease update: bacterial spot of peach and other stone fruit",
        "source_url": "https://extension.psu.edu/2026-disease-update-bitter-rot-and-summer-disease-season",
        "source_publisher": "Penn State Extension",
        "text": (
            "Bacterial spot can affect peach and other stone fruit. Leaf symptoms may include "
            "water-soaked lesions that become brown and may leave small holes; fruit can develop "
            "sunken spots. Symptoms vary with cultivar and weather and can resemble other "
            "problems. Confirm the cause with a local fruit-tree specialist and follow current "
            "regional guidance; do not choose a pesticide from an image prediction alone."
        ),
    },
    {
        "id": "PV-STRAWBERRY-LEAF-SCORCH-01",
        "title": "Strawberry leaf scorch: symptoms and prevention",
        "source": "UC Statewide IPM Program",
        "source_title": "Leaf scorch of strawberries",
        "source_url": "https://ipm.ucanr.edu/home-and-landscape/leaf-scorch-of-strawberries/",
        "source_publisher": "UC Statewide IPM Program",
        "text": (
            "Strawberry leaf scorch may cause brown-to-purple spots with indistinct borders. "
            "As disease increases, tissue between lesions may redden or turn purple, then leaves "
            "dry and curl at the margins. Common leaf spot and other diseases can look similar. "
            "Reduce crowding, improve airflow, remove infected debris where appropriate, and "
            "use drip or furrow irrigation where practical to keep foliage drier."
        ),
    },
    {
        "id": "PV-TOMATO-LEAF-MOLD-01",
        "title": "Tomato leaf mold: symptoms and prevention",
        "source": "University of Minnesota Extension",
        "source_title": "Tomato leaf mold",
        "source_url": "https://extension.umn.edu/agriculture/specialty-crops/vegetable-farming/disease-management/tomato-leaf-mold",
        "source_publisher": "University of Minnesota Extension",
        "text": (
            "Tomato leaf mold often affects older leaves first, with pale yellow spots on the "
            "upper surface and olive-green to brown velvety growth on the underside. High "
            "relative humidity, especially in greenhouses and high tunnels, favors the disease. "
            "Improve ventilation and airflow, avoid excessive canopy density, remove infected "
            "plant debris, and consider resistant cultivars. Confirm the diagnosis before "
            "selecting disease-specific treatment."
        ),
    },
    {
        "id": "PV-TOMATO-SEPTORIA-01",
        "title": "Tomato Septoria leaf spot: symptoms and prevention",
        "source": "University of Maryland Extension",
        "source_title": "Septoria leaf spot of tomatoes",
        "source_url": "https://www.extension.umd.edu/resource/septoria-leaf-spot-tomatoes",
        "source_publisher": "University of Maryland Extension",
        "text": (
            "Tomato Septoria leaf spot often starts on lower leaves as small circular gray "
            "lesions with dark borders; tiny black fungal structures may be visible inside "
            "spots. Wet weather and splashing water favor spread. Increase spacing and airflow, "
            "water near the soil rather than wetting foliage, mulch to reduce soil splash, and "
            "remove affected leaves where appropriate. Similar leaf spots need confirmation."
        ),
    },
    {
        "id": "PV-TOMATO-TARGET-SPOT-01",
        "title": "Tomato target spot: symptoms and prevention",
        "source": "University of Florida IFAS Extension",
        "source_title": "Target spot of tomato in Florida",
        "source_url": "https://ask.ifas.ufl.edu/publication/PP351",
        "source_publisher": "University of Florida IFAS Extension",
        "text": (
            "Tomato target spot can cause dark-brown pinpoint leaf lesions that enlarge, "
            "develop lighter centers and concentric dark rings, and sometimes have yellowing "
            "around them. Fruit lesions may become sunken and deeply pitted. It can resemble "
            "early blight or bacterial spot, and prolonged leaf wetness favors infection. "
            "Use clean transplants, manage weeds and plant debris, and consult local extension "
            "guidance before selecting treatment."
        ),
    },
    {
        "id": "PV-INDIA-ICAR-IPM-RESOURCES-01",
        "title": "India crop pest and disease management resources from ICAR NRIIPM",
        "source": "ICAR-National Research Institute for Integrated Pest Management",
        "source_title": "Extension Folders",
        "source_url": "https://nriipm.res.in/extension-folders.aspx",
        "source_publisher": "Indian Council of Agricultural Research (ICAR), NRIIPM",
        "text": (
            "For India-specific crop protection guidance, ICAR-NRIIPM publishes extension "
            "folders on management of major pests and diseases for crops including bitter "
            "gourd, bottle gourd, brinjal, chilli, cole crops, cucumber, guava, kinnow, okra, "
            "onion, potato and tomato. The site also lists resources on maize pest and disease "
            "identification, integrated pest management in basmati rice, and cotton boll rot. "
            "Use these crop-specific extension materials to cross-check local symptoms and "
            "management advice. Availability of a resource does not mean it applies to every "
            "state, cultivar, season or production system; consult local agricultural experts."
        ),
    },
    {
        "id": "PV-INDIA-ICAR-SURVEILLANCE-01",
        "title": "Indian crop pest and disease surveillance databases",
        "source": "ICAR-National Research Institute for Integrated Pest Management",
        "source_title": "Online Databases",
        "source_url": "https://nriipm.res.in/OnlineDatabases.aspx",
        "source_publisher": "Indian Council of Agricultural Research (ICAR), NRIIPM",
        "text": (
            "ICAR-NRIIPM lists agricultural surveillance and decision-support resources, "
            "including the National Pest Surveillance System, the Horticulture Pest Surveillance "
            "and Advisory Project for Maharashtra, Haryana horticultural pest surveillance and "
            "advisory services, a mango decision-support system, and a crop pest information "
            "system. These resources can help users look for region- and crop-relevant alerts "
            "or advisories. Check the linked service for current coverage and recommendations; "
            "this knowledge passage is a directory summary, not a live surveillance result."
        ),
    },
    {
        "id": "PV-INDIA-ICAR-IIHR-PORTAL-01",
        "title": "Indian horticulture crop diagnosis and management portal",
        "source": "ICAR-Indian Institute of Horticultural Research",
        "source_title": "Web applications on horticultural crops",
        "source_url": "https://webapps.iihr.res.in/",
        "source_publisher": "Indian Council of Agricultural Research (ICAR), IIHR",
        "text": (
            "ICAR-IIHR provides web applications for fruit and vegetable crop management, "
            "including disease management and pest management modules with symptom information "
            "and control measures. The portal lists crop resources such as mango, papaya, guava, "
            "pomegranate, sapota, watermelon, okra, brinjal, onion, grapes, chilli and tomato. "
            "Use the portal to find crop-specific information and cross-check a model prediction. "
            "A resource listing is not itself a diagnosis; verify the crop, disease, region and "
            "current guidance before acting."
        ),
    },
    {
        "id": "PV-INDIA-ICAR-KNOWLEDGE-PORTAL-01",
        "title": "ICAR agricultural knowledge and crop expert resources",
        "source": "Indian Council of Agricultural Research",
        "source_title": "Technologies and Knowledge Resources",
        "source_url": "https://icar.gov.in/en/technologies-and-knowledge-resources",
        "source_publisher": "Indian Council of Agricultural Research (ICAR)",
        "text": (
            "ICAR maintains a directory of agricultural digital resources, including crop "
            "knowledge portals and expert systems such as rice knowledge management, maize "
            "expert systems, wheat crop management, seed-spice expert systems and CaneInfo. "
            "These can be starting points for crop-specific background and management information. "
            "Open the original resource to check its scope, date and geographic relevance. "
            "This directory does not replace current state agricultural university or extension advice."
        ),
    }

    {
        "id": "PV-DIAG-DIFFERENTIAL-01",
        "title": "Plant problem diagnosis: compare biological and environmental causes",
        "source": "UC Statewide IPM Program",
        "source_title": "Diagnosing the Causes of Unhealthy Plants",
        "source_url": "https://ipm.ucanr.edu/agriculture/floriculture-and-ornamental-nurseries/diagnosing-the-causess-of-unhealthy-plants/",
        "source_publisher": "University of California Statewide Integrated Pest Management Program",
        "text": (
            "A reliable plant-health assessment compares living causes (fungi, bacteria, "
            "viruses, insects, mites and nematodes) with nonliving causes such as heat or cold, "
            "water stress, poor drainage, nutrient imbalance, salinity, chemical injury and "
            "mechanical damage. Record which plant parts are affected, whether symptoms appear "
            "on one plant or many, their distribution in the field, the age of affected leaves, "
            "recent weather and recent irrigation, fertilizer or spray changes. Look for direct "
            "signs such as insects, webbing, fungal growth or bacterial ooze, not symptoms alone. "
            "More than one cause can occur at once. An image prediction is a hypothesis, not a "
            "confirmed diagnosis; use a local diagnostic service when the result will guide costly "
            "or high-impact treatment."
        ),
    },
    {
        "id": "PV-ABIOTIC-STRESS-01",
        "title": "Yellow leaves, spots and wilting: check abiotic stress and root health",
        "source": "UC Statewide IPM Program",
        "source_title": "Nutrient Disorders",
        "source_url": "https://ipm.ucanr.edu/home-and-landscape/nutrient-disorders/",
        "source_publisher": "University of California Statewide Integrated Pest Management Program",
        "text": (
            "Yellowing, leaf-edge burn, spotting, distortion, stunting and dieback are not "
            "specific to infectious disease. Potential causes include overwatering or drought, "
            "poor drainage and damaged roots, unsuitable soil pH, salinity, nutrient deficiency "
            "or excess, temperature extremes and chemical injury. Nutrients can be present in "
            "soil but unavailable to roots because of pH or root-zone problems; excess of one "
            "nutrient can also interfere with another. Compare old and new leaves, check soil "
            "moisture and drainage, review recent fertilizer or spray use, and consider soil or "
            "plant-tissue testing before adding fertilizer. Do not diagnose a nutrient deficiency "
            "from leaf color alone or prescribe fertilizer amounts from an image."
        ),
    },
    {
        "id": "PV-IPM-FOUNDATIONS-01",
        "title": "Integrated pest management for Indian crops",
        "source": "ICAR-National Research Institute for Integrated Pest Management",
        "source_title": "Technical Bulletins",
        "source_url": "https://nriipm.res.in/technical-bulletins.aspx",
        "source_publisher": "Indian Council of Agricultural Research (ICAR), NRIIPM",
        "text": (
            "Integrated pest management (IPM) combines field monitoring, correct identification, "
            "prevention, cultural practices, conservation of beneficial organisms and carefully "
            "selected controls when justified. ICAR-NRIIPM lists technical resources for crops "
            "including cotton, basmati rice, mustard and horticultural crops, as well as pest "
            "surveillance and IPM strategies. The suitable approach depends on crop, growth stage, "
            "region, season and confirmed pest or disease. Prefer current crop- and location-specific "
            "recommendations over a generic answer. This directory summary is not a current pest "
            "alert and does not establish a treatment threshold for a particular field."
        ),
    },
    {
        "id": "PV-DISEASE-SANITATION-01",
        "title": "Disease prevention through scouting, sanitation and moisture management",
        "source": "University of Minnesota Extension",
        "source_title": "Managing plant diseases in the home garden",
        "source_url": "https://extension.umn.edu/garden-and-home/yard-and-garden/yard-and-garden-problems/managing-plant-diseases-in-the-home-garden",
        "source_publisher": "University of Minnesota Extension",
        "text": (
            "Scout crops regularly and note whether symptoms are new, stable or spreading. "
            "Many fungal and bacterial problems are favored by prolonged leaf wetness and can "
            "spread through splashing water or handling wet plants. Where appropriate, water near "
            "the root zone, allow foliage to dry, maintain suitable airflow, clean tools and avoid "
            "moving from affected plants to healthy ones without hygiene. Remove infected debris "
            "only in ways appropriate for the crop and disease; some infected material should not "
            "be composted or transported. Crop rotation is useful only when the pathogen host range "
            "and production system make it effective. These practices reduce risk but do not prove "
            "which disease is present."
        ),
    },
    {
        "id": "PV-INDIA-MAHARASHTRA-SURVEILLANCE-01",
        "title": "India and Maharashtra crop pest surveillance resources",
        "source": "ICAR-National Research Institute for Integrated Pest Management",
        "source_title": "Technical Bulletins",
        "source_url": "https://nriipm.res.in/technical-bulletins.aspx",
        "source_publisher": "Indian Council of Agricultural Research (ICAR), NRIIPM",
        "text": (
            "ICAR-NRIIPM publishes resources describing horticultural pest surveillance and "
            "advisory systems, including the Crop Pest Surveillance and Advisory Project (CROPSAP) "
            "of Maharashtra, and manuals for surveillance of major horticultural crops. For a farm "
            "in Maharashtra, check the current state or district agricultural advisory and the "
            "official service's latest bulletin for crop, pest, location and date. This passage "
            "only identifies a source to consult; it does not contain live surveillance data, a "
            "current outbreak alert or a recommendation that a particular pest is present."
        ),
    },
    {
        "id": "PV-NUTRIENT-DIFFERENTIAL-01",
        "title": "Nutrient deficiency symptoms require confirmation",
        "source": "UC Statewide IPM Program",
        "source_title": "Nutrient Deficiencies",
        "source_url": "https://ipm.ucanr.edu/home-and-landscape/nutrient-deficiencies/",
        "source_publisher": "University of California Statewide Integrated Pest Management Program",
        "text": (
            "Nutrient problems can cause discoloration, smaller leaves, distorted growth, fewer "
            "flowers or fruit, stunting and dieback. Some patterns can guide investigation: older "
            "leaves may become uniformly pale with nitrogen shortage, while iron-related chlorosis "
            "often affects newer leaves with greener veins. However, symptoms vary by species and "
            "can be caused by root damage, soil pH, salinity, excess water, drought or disease. "
            "Use soil and, where appropriate, plant-tissue testing and crop-specific recommendations "
            "before correcting a suspected deficiency. Avoid adding fertilizer solely because an "
            "image model reports yellow leaves."
        ),
    },
    {
        "id": "PV-TREATMENT-SAFETY-01",
        "title": "Safe and responsible crop protection decisions",
        "source": "ICAR-National Research Institute for Integrated Pest Management",
        "source_title": "Technical Bulletins",
        "source_url": "https://nriipm.res.in/technical-bulletins.aspx",
        "source_publisher": "Indian Council of Agricultural Research (ICAR), NRIIPM",
        "text": (
            "A plant image classifier cannot determine the correct pesticide, formulation, dose, "
            "mixing rate, pre-harvest interval or legal approval for a particular field. Confirm "
            "the pest or disease, crop and growth stage, then consult current local agricultural "
            "extension guidance and the product label approved for that crop and use. Follow label "
            "directions, personal protective equipment, re-entry and harvest restrictions, and "
            "environmental precautions. Consider nonchemical and cultural IPM options where suitable "
            "and protect pollinators and beneficial organisms. This knowledge base intentionally "
            "does not provide pesticide dosage instructions; never infer a dose from a model result."
        ),
    },
    {
        "id": "PV-FIELD-OBSERVATION-01",
        "title": "Field observation checklist before interpreting a plant image",
        "source": "UC Statewide IPM Program",
        "source_title": "Diagnosing Plant Problems",
        "source_url": "https://ipm.ucanr.edu/agriculture/floriculture-and-ornamental-nurseries/diagnosing-plant-problems/",
        "source_publisher": "University of California Statewide Integrated Pest Management Program",
        "text": (
            "Improve a plant-health assessment by recording the crop and variety if known, plant "
            "age or growth stage, affected plant part, whether symptoms begin on old or new leaves, "
            "the number and distribution of affected plants, recent rainfall and temperature, "
            "irrigation and drainage, fertilizer and spray history, nearby weeds or alternate hosts, "
            "and visible insects, webbing, spores or lesions. Photograph the whole plant and the "
            "symptom close-up in good light, including healthy comparison leaves when possible. "
            "A single leaf photo may omit important context and look similar across diseases and "
            "abiotic disorders. Use these observations to guide follow-up questions, not to claim "
            "laboratory confirmation."
        ),
    },

]

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in",
    "is", "it", "of", "on", "or", "the", "to", "with", "what", "how",
    "plant", "plants", "disease", "diseases", "symptom", "symptoms",
    "prevention", "prevent", "treatment", "treatments", "management",
    "guidance", "general", "common", "check", "checks", "leaf", "leaves",
    "crop", "crops", "care", "condition", "conditions", "help", "advice", "two", "spotted",
}
_TOKEN_ALIASES = {
    "viral": "virus",
    "viruses": "virus",
    "blights": "blight",
    "spots": "spot",
    "lesions": "lesion",
    "leaves": "leaf",
    "mites": "mite",
}


def _tokens(text):
    """Normalize words and common label variants for deterministic retrieval."""
    raw_tokens = re.findall(r"[a-z0-9]+", str(text).lower())
    normalized = [_TOKEN_ALIASES.get(token, token) for token in raw_tokens]
    return [token for token in normalized if token not in _STOPWORDS]


_EMBEDDING_CACHE = {}


def _retrieve_lexical(query, top_k):
    """Rank source passages using crop-aware lexical matching and IDF weighting."""
    query_tokens = list(dict.fromkeys(_tokens(query)))
    if not query_tokens:
        return []

    query_set = set(query_tokens)
    if "healthy" in query_set:
        general = next((doc for doc in KNOWLEDGE_DOCUMENTS if doc["id"] == "PV-GENERAL-01"), None)
        return [{**general, "score": 1.0, "retrieval_method": "lexical"}] if general else []

    crop_terms = {
        "apple", "blueberry", "cherry", "corn", "maize", "grape", "orange", "citrus",
        "peach", "pepper", "potato", "raspberry", "soybean", "squash", "strawberry", "tomato"
    }
    query_crops = query_set & crop_terms
    corpus_size = max(len(KNOWLEDGE_DOCUMENTS), 1)
    tokenized_docs = []
    for doc in KNOWLEDGE_DOCUMENTS:
        title = set(_tokens(doc["title"]))
        body = set(_tokens(doc["text"]))
        doc_crops = set(_tokens(doc["title"] + " " + doc["text"])) & crop_terms
        tokenized_docs.append((doc, title, body, doc_crops))
    document_frequency = {token: sum(token in title or token in body for _, title, body, _ in tokenized_docs) for token in query_set}
    scored = []
    for doc, title_tokens, body_tokens, doc_crops in tokenized_docs:
        matched = query_set & (title_tokens | body_tokens)
        if not matched:
            continue
        # When a crop is explicitly named, strongly prefer that crop-specific evidence.
        if query_crops and doc_crops and not (query_crops & doc_crops):
            crop_penalty = 0.12
        else:
            crop_penalty = 1.0
        weighted_score = 0.0
        for token in matched:
            idf = 1.0 + math.log((corpus_size + 1) / (document_frequency[token] + 1))
            weighted_score += (4.0 if token in title_tokens else 1.0) * idf
        coverage = len(matched) / len(query_set)
        # Require more than a generic shared disease word when the crop is known.
        crop_match = bool(query_crops & doc_crops) if query_crops else False
        score = weighted_score * (0.12 + 0.88 * coverage) * crop_penalty
        if query_crops and not crop_match:
            score *= 0.35
        scored.append((score, doc))
    scored.sort(key=lambda item: (item[0], item[1]["id"]), reverse=True)
    return [{**doc, "score": round(score, 4), "retrieval_method": "lexical"} for score, doc in scored[:top_k]]

def _retrieve_semantic(query, top_k, provider, api_key):
    import hashlib
    from rag_embeddings import cosine_similarity, embed_texts

    document_texts = [
        f"{doc['title']}\n{doc['text']}"
        for doc in KNOWLEDGE_DOCUMENTS
    ]
    corpus_hash = hashlib.sha256(
        "\n".join(document_texts).encode("utf-8")
    ).hexdigest()
    key_fingerprint = hashlib.sha256(str(api_key).encode("utf-8")).hexdigest()
    cache_key = (provider, key_fingerprint, corpus_hash)

    if cache_key not in _EMBEDDING_CACHE:
        _EMBEDDING_CACHE[cache_key] = embed_texts(provider, api_key, document_texts)
    document_vectors = _EMBEDDING_CACHE[cache_key]
    query_vector = embed_texts(provider, api_key, [query])[0]

    scored = []
    for doc, vector in zip(KNOWLEDGE_DOCUMENTS, document_vectors):
        score = cosine_similarity(query_vector, vector)
        if score >= 0.35:
            scored.append((score, doc))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [
        {
            **doc,
            "score": round(score, 4),
            "retrieval_method": "semantic",
        }
        for score, doc in scored[:top_k]
    ]


def retrieve_knowledge(query, top_k=3, provider=None, api_key=None):
    """Retrieve source-backed passages with optional semantic + lexical rank fusion.

    Set PLANTVISION_SEMANTIC_RETRIEVAL=0 to avoid embedding API calls and use
    the deterministic lexical ranker only. Semantic failures fall back safely.
    """
    import os
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer")

    lexical = _retrieve_lexical(query, max(top_k, top_k * 2))
    semantic_enabled = os.getenv("PLANTVISION_SEMANTIC_RETRIEVAL", "0").strip().lower() not in {"0", "false", "no", "off"}
    if not semantic_enabled or not provider or not api_key:
        return lexical[:top_k]

    try:
        semantic = _retrieve_semantic(query, max(top_k, top_k * 2), provider, api_key)
        if not semantic:
            return [{**doc, "retrieval_method": "lexical_fallback"} for doc in lexical[:top_k]]

        # Reciprocal Rank Fusion combines independent lexical and semantic rankings
        # without assuming their raw scores are calibrated to the same scale.
        fused = {}
        by_id = {}
        for rank, doc in enumerate(lexical, start=1):
            doc_id = doc["id"]
            by_id[doc_id] = doc
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (60 + rank)
        for rank, doc in enumerate(semantic, start=1):
            doc_id = doc["id"]
            by_id[doc_id] = doc
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (60 + rank)
        ranked = sorted(fused, key=lambda doc_id: (fused[doc_id], doc_id), reverse=True)
        results = []
        for doc_id in ranked[:top_k]:
            doc = dict(by_id[doc_id])
            doc["score"] = round(fused[doc_id], 5)
            doc["retrieval_method"] = "hybrid_rrf"
            results.append(doc)
        return results
    except Exception:
        # Keep the assistant usable when embeddings hit a quota or provider outage.
        return [{**doc, "retrieval_method": "lexical_fallback"} for doc in lexical[:top_k]]
