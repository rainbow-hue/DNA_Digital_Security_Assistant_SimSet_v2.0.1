# -*- coding: utf-8 -*-
"""Static knowledge for the assistant: everything here is either loaded from
the package's own data/reference files or is user-facing wording (fi/en/sv)
written to match the package's step / topic regexes.

Nothing in this file is learned; the learned parts live in model.py/train.py.
"""
import json
import os
import re

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
REF = os.path.join(ROOT, "data", "reference")
LANGS = ("fi", "en", "sv")
GATED = "available_with_consent"


def _load(name):
    with open(os.path.join(REF, name), encoding="utf-8") as f:
        return json.load(f)


ACTION_VOCAB = _load("action_vocabulary.json")
ACTIONS = list(ACTION_VOCAB["actions"])
INTERVENTION = {k: v["intervention"] for k, v in ACTION_VOCAB["actions"].items()}
RISKS = list(ACTION_VOCAB["risk_levels"])                       # none..critical
URGENCIES = list(ACTION_VOCAB["urgency"])                       # none, routine, today, immediate
ESC_TARGETS = list(ACTION_VOCAB["escalation_targets"])
STEP_IDS = list(_load("step_vocabulary.json")["steps"])
TOPIC_IDS = list(_load("question_topics.json")["topics"])
SIGNAL_DICT = _load("signal_dictionary.json")["signals"]
SENSITIVE_PATHS = [k for k, v in SIGNAL_DICT.items() if v.get("consent") == "requires_consent"]

with open(os.path.join(REF, "guidance_corpus.jsonl"), encoding="utf-8") as f:
    GUIDANCE = [json.loads(l) for l in f if l.strip()]
GUIDANCE_BY_ID = {g["id"]: g for g in GUIDANCE}


# ---------------------------------------------------------------------------
# 1. User-facing step sentences. Every sentence is checked by
#    tools in train.py (self_test) against data/reference/step_vocabulary.json
#    so the evaluator recognises it, and against unsafe_claims.json.
# ---------------------------------------------------------------------------
STEP_TEXT = {
    "dont_open_link": {
        "en": "Do not open the link in the message.",
        "fi": "Älä avaa viestissä olevaa linkkiä.",
        "sv": "Öppna inte länken i meddelandet."},
    "dont_pay_or_transfer": {
        "en": "Do not pay or send any money before the matter has been checked.",
        "fi": "Älä maksa äläkä siirrä rahaa ennen kuin asia on varmistettu.",
        "sv": "Betala inte och för inte över pengar innan saken är kontrollerad."},
    "dont_reply": {
        "en": "Do not reply to the message.",
        "fi": "Älä vastaa viestiin.",
        "sv": "Svara inte på meddelandet."},
    "delete_or_report_message": {
        "en": "You can delete the message or report it as spam.",
        "fi": "Voit poistaa viestin tai ilmoittaa sen roskapostiksi.",
        "sv": "Du kan radera meddelandet eller rapportera det som skräppost."},
    "verify_official_channel": {
        "en": "Check the matter yourself in the organisation's own app or on its official website.",
        "fi": "Tarkista asia itse palvelun omasta sovelluksesta tai virallisilta verkkosivuilta.",
        "sv": "Kontrollera saken själv i tjänstens egen app eller på den officiella webbplatsen."},
    "verify_out_of_band": {
        "en": "Contact the person or company on a number you already know, not one given in the message.",
        "fi": "Ota yhteyttä numeroon, jonka tiedät jo oikeaksi, äläkä käytä viestissä annettua numeroa.",
        "sv": "Kontakta personen eller företaget på ett nummer som du redan känner."},
    "contact_bank_now": {
        "en": "Contact your bank right away and tell them what happened.",
        "fi": "Ota heti yhteyttä pankkiisi ja kerro, mitä tapahtui.",
        "sv": "Kontakta din bank direkt och berätta vad som hände."},
    "block_card": {
        "en": "Have the bank close the card right away.",
        "fi": "Pyydä pankkia sulkemaan kortti heti.",
        "sv": "Be banken att frysa ditt kort direkt."},
    "change_password": {
        "en": "Change the password of the affected account now.",
        "fi": "Vaihda kyseisen palvelun salasana nyt.",
        "sv": "Byt lösenordet för det berörda kontot nu."},
    "enable_mfa": {
        "en": "Turn on two-step verification for the account.",
        "fi": "Ota käyttöön kaksivaiheinen tunnistautuminen.",
        "sv": "Aktivera tvåstegsverifiering för kontot."},
    "use_trusted_device": {
        "en": "Do this on another device that you trust.",
        "fi": "Tee tämä luotettavalla laitteella.",
        "sv": "Gör det på en annan enhet som du litar på."},
    "check_password_reuse": {
        "en": "If you use the same password anywhere else, change it there too.",
        "fi": "Jos käytät samaa salasanaa muissa palveluissa, vaihda se myös niissä.",
        "sv": "Om du använder samma lösenord i andra tjänster, byt det där också."},
    "review_sessions": {
        "en": "Check recent logins and sign out of all devices you do not recognise.",
        "fi": "Tarkista viimeaikaiset kirjautumiset ja kirjaudu ulos tuntemattomilta laitteilta.",
        "sv": "Kontrollera senaste inloggningar och logga ut från okända enheter."},
    "dont_approve_prompts": {
        "en": "Do not approve any login request that you did not start yourself.",
        "fi": "Älä hyväksy kirjautumispyyntöjä, joita et itse aloittanut.",
        "sv": "Godkänn aldrig inloggningsförfrågningar som du inte själv har startat."},
    "dont_share_codes": {
        "en": "Never give a code or PIN to anyone who calls or messages you.",
        "fi": "Älä kerro koodeja tai tunnuslukuja kenellekään.",
        "sv": "Dela aldrig koder eller PIN-koder med någon."},
    "disconnect_device": {
        "en": "Disconnect the device from the internet now, for example by turning on flight mode.",
        "fi": "Katkaise laitteen verkkoyhteys nyt, esimerkiksi laittamalla lentotila päälle.",
        "sv": "Koppla bort enheten från internet nu, till exempel med flygläge."},
    "remove_app_or_profile": {
        "en": "Remove the app, profile or program that was installed.",
        "fi": "Poista asennettu sovellus, profiili tai ohjelma.",
        "sv": "Ta bort appen, profilen eller programmet som installerades."},
    "get_device_checked": {
        "en": "Have the device checked with a security scan or by a technical expert.",
        "fi": "Tarkistuta laite tietoturvaohjelmalla tai asiantuntijalla.",
        "sv": "Låt en expert eller ett säkerhetsprogram kontrollera enheten."},
    "avoid_banking_on_device": {
        "en": "Do not use online banking or other important logins on it until it has been checked.",
        "fi": "Älä käytä laitteella verkkopankkia ennen kuin laite on tarkistettu.",
        "sv": "Använd inte internetbanken på enheten innan den är kontrollerad."},
    "contact_operator": {
        "en": "Contact the operator's customer service right away.",
        "fi": "Ota heti yhteyttä operaattorin asiakaspalveluun.",
        "sv": "Kontakta operatörens kundtjänst direkt."},
    "lock_subscription": {
        "en": "Ask them to lock your number and stop any SIM or eSIM changes.",
        "fi": "Pyydä heitä lukitsemaan numerosi ja estämään SIM- ja eSIM-muutokset.",
        "sv": "Be dem spärra ditt nummer och stoppa alla SIM- och eSIM-ändringar."},
    "report_police": {
        "en": "Report the crime to the police.",
        "fi": "Tee rikosilmoitus poliisille.",
        "sv": "Gör en polisanmälan."},
    "report_cyber_authority": {
        "en": "You can also report it to the National Cyber Security Centre.",
        "fi": "Voit ilmoittaa asiasta myös Kyberturvallisuuskeskukselle.",
        "sv": "Du kan också rapportera det till Cybersäkerhetscentret."},
    "keep_evidence": {
        "en": "Keep the messages and screenshots as evidence.",
        "fi": "Ota viestit ja kuvakaappaukset talteen todisteiksi.",
        "sv": "Spara meddelandena och skärmdumpar som bevis."},
    "hang_up_dont_call_back": {
        "en": "Hang up and do not call the number back.",
        "fi": "Lopeta puhelu äläkä soita numeroon takaisin.",
        "sv": "Lägg på och ring inte tillbaka."},
    "two_person_approval": {
        "en": "From now on, use two-person approval for payments and bank detail changes.",
        "fi": "Ottakaa käyttöön kahden henkilön hyväksyntä maksuille ja tilinumeromuutoksille.",
        "sv": "Låt två personer godkänna betalningar och ändringar av kontonummer."},
    "revoke_access": {
        "en": "Remove access for anyone who should no longer have it.",
        "fi": "Poistakaa käyttöoikeudet ihmisiltä, joilla ei enää pidä olla pääsyä.",
        "sv": "Ta bort åtkomst för personer som inte längre ska ha den."},
    "check_forwarding_rules": {
        "en": "Check the mailbox for forwarding and inbox rules that you did not set.",
        "fi": "Tarkistakaa sähköpostin edelleenlähetykset ja postilaatikon säännöt.",
        "sv": "Kontrollera vidarebefordringar och inkorgsregler i e-posten."},
    "inform_staff": {
        "en": "Warn your staff and any customers who may have received fake messages.",
        "fi": "Varoittakaa työntekijöitä ja asiakkaita, jotka ovat voineet saada huijausviestejä.",
        "sv": "Varna personal och kunder som kan ha fått falska meddelanden."},
    "dont_open_attachment": {
        "en": "Do not open the attachment, and do not allow any content in it.",
        "fi": "Älä avaa liitettä äläkä hyväksy siinä mitään sisältöä.",
        "sv": "Öppna inte bilagan och tillåt inget innehåll i den."},
    "secure_router": {
        "en": "Change the router admin password and update the router software.",
        "fi": "Vaihda reitittimen hallintasalasana ja päivitä reitittimen ohjelmisto.",
        "sv": "Byt routerns adminlösenord och uppdatera routerns programvara."},
    "explain_benign": {
        "en": "This looks like a normal, genuine message.",
        "fi": "Tämä vaikuttaa tavalliselta ja aidolta viestiltä.",
        "sv": "Det här verkar äkta och normalt, och du behöver inte göra något."},
    "seek_support_org": {
        "en": "Talk to someone you trust or a victim support service.",
        "fi": "Ota yhteyttä tukipalveluun, kuten Rikosuhripäivystykseen, tai puhu ihmiselle, johon luotat.",
        "sv": "Prata med någon du litar på eller med Brottsofferjouren."},
    "plan_before_removing": {
        "en": "Plan your safety before removing anything, because removing it may alert the other person.",
        "fi": "Mieti turvallisuuttasi ennen poistamista, koska poisto voi paljastaa sen toiselle.",
        "sv": "Planera din säkerhet innan du tar bort något, eftersom det kan varna den andra personen."},
    "notify_data_protection": {
        "en": "If customers' personal data may have leaked, report it to the Data Protection Ombudsman within 72 hours.",
        "fi": "Jos asiakkaiden henkilötietoja on voinut vuotaa, ilmoittakaa siitä tietosuojavaltuutetulle 72 tunnin kuluessa.",
        "sv": "Om kunders personuppgifter kan ha läckt, anmäl det till dataombudsmannen inom 72 timmar."},
    "decline_request": {
        "en": "I can't help find out who is behind the number or share their details.",
        "fi": "En voi selvittää, kuka numeron takana on, enkä jakaa hänen tietojaan.",
        "sv": "Jag kan inte hjälpa till att ta reda på vem som står bakom numret."},
    "check_statements": {
        "en": "Check your bank transactions for any payments you do not recognise.",
        "fi": "Tarkista tilitapahtumat tuntemattomien veloitusten varalta.",
        "sv": "Kontrollera dina transaktioner efter okända debiteringar."},
    "update_device": {
        "en": "Update the device software.",
        "fi": "Päivitä laitteen ohjelmisto.",
        "sv": "Uppdatera enhetens programvara."},
}

# Order in which steps are written (most urgent first).
STEP_PRIORITY = [
    "decline_request", "plan_before_removing", "seek_support_org", "explain_benign",
    "disconnect_device", "contact_bank_now", "block_card", "contact_operator", "lock_subscription",
    "hang_up_dont_call_back", "dont_share_codes", "dont_approve_prompts", "dont_pay_or_transfer",
    "dont_open_link", "dont_open_attachment", "dont_reply", "change_password", "use_trusted_device",
    "check_password_reuse", "enable_mfa", "review_sessions", "check_forwarding_rules", "revoke_access",
    "remove_app_or_profile", "avoid_banking_on_device", "get_device_checked", "secure_router",
    "update_device", "verify_out_of_band", "verify_official_channel", "check_statements",
    "inform_staff", "two_person_approval", "notify_data_protection", "report_police",
    "report_cyber_authority", "keep_evidence", "delete_or_report_message",
]

# Steps that must appear for an action (safety floor, independent of the model).
ACTION_CORE_STEPS = {
    "no_action_needed": ["explain_benign"],
    "do_not_interact_and_delete": [],
    "verify_via_official_channel": ["verify_official_channel"],
    "contact_operator_support": ["contact_operator"],
    "change_password_and_enable_mfa": ["change_password"],
    "isolate_device_and_scan": ["disconnect_device"],
    "contact_bank_and_block_cards": ["contact_bank_now"],
    "report_to_authority": ["report_police"],
    "escalate_to_human": [],
    "decline_out_of_scope": ["decline_request"],
}

# Second sentence: the short "why" (plain language, no jargon).
WHY = {
    "no_action_needed": {
        "en": "It does not ask you for codes, passwords or payments, and it matches what you expected.",
        "fi": "Siinä ei pyydetä koodeja, salasanoja tai maksuja, ja se vastaa sitä, mitä odotit.",
        "sv": "Det ber inte om koder, lösenord eller betalningar."},
    "do_not_interact_and_delete": {
        "en": "Messages like this are a common scam that tries to get money or card details.",
        "fi": "Tällaiset viestit ovat yleinen huijaus, jolla yritetään saada rahaa tai korttitietoja.",
        "sv": "Sådana meddelanden är ett vanligt bedrägeri som vill komma åt pengar eller kortuppgifter."},
    "verify_via_official_channel": {
        "en": "Scammers often copy real companies, so the safest way is to check through a channel you open yourself.",
        "fi": "Huijarit matkivat usein oikeita yrityksiä, joten turvallisinta on tarkistaa asia itse avaamasi kanavan kautta.",
        "sv": "Bedragare kopierar ofta riktiga företag, så det säkraste är att kontrollera via en kanal du själv öppnar."},
    "contact_operator_support": {
        "en": "Unexpected changes to your SIM, eSIM or number can be used to take over your number, so they need to be stopped quickly.",
        "fi": "Odottamattomilla SIM-, eSIM- tai numeromuutoksilla voidaan kaapata numerosi, joten ne pitää pysäyttää nopeasti.",
        "sv": "Oväntade ändringar av SIM, eSIM eller nummer kan användas för att ta över ditt nummer."},
    "change_password_and_enable_mfa": {
        "en": "Someone else may know or try your password, and a new one keeps them out.",
        "fi": "Joku muu voi tietää salasanasi tai yrittää käyttää sitä, ja uusi salasana pitää hänet ulkona.",
        "sv": "Någon annan kan känna till eller prova ditt lösenord, och ett nytt håller dem ute."},
    "isolate_device_and_scan": {
        "en": "Software installed this way can watch what you do and steal banking details.",
        "fi": "Tällä tavalla asennettu ohjelma voi seurata tekemisiäsi ja varastaa pankkitietoja.",
        "sv": "Program som installerats så här kan se vad du gör och stjäla bankuppgifter."},
    "contact_bank_and_block_cards": {
        "en": "Acting quickly gives the bank the best chance to stop or recover the money.",
        "fi": "Nopea toiminta antaa pankille parhaat mahdollisuudet pysäyttää tai palauttaa rahat.",
        "sv": "Snabb handling ger banken bäst chans att stoppa eller få tillbaka pengarna."},
    "report_to_authority": {
        "en": "A report helps the police investigate and is often needed for the bank.",
        "fi": "Ilmoitus auttaa poliisia selvittämään asiaa, ja pankki tarvitsee sen usein.",
        "sv": "En anmälan hjälper polisen och behövs ofta av banken."},
    "escalate_to_human": {
        "en": "This is a situation where a trained person can help you more safely than a chat can.",
        "fi": "Tässä tilanteessa koulutettu ihminen auttaa sinua turvallisemmin kuin chat.",
        "sv": "I den här situationen kan en utbildad person hjälpa dig säkrare än en chatt."},
    "decline_out_of_scope": {
        "en": "Tracing people yourself can be risky and can hurt innocent people.",
        "fi": "Ihmisten jäljittäminen omin päin voi olla vaarallista ja osua syyttömiin.",
        "sv": "Att själv försöka spåra personer kan vara riskabelt och drabba oskyldiga."},
}

WHY_CALL = {
    "en": "Calls like this are a common scam that tries to get money or information from you.",
    "fi": "Tällaiset puhelut ovat yleinen huijaus, jolla yritetään saada sinulta rahaa tai tietoja.",
    "sv": "Sådana samtal är ett vanligt bedrägeri som vill komma åt pengar eller uppgifter.",
}

# Hand-off sentences (deliberately avoid phrasing that the step regexes would
# read as an extra, unrelated step).
ESCALATION_TEXT = {
    "operator_support_agent": {
        "en": "I can hand this over to our customer service agent together with a summary of what we know.",
        "fi": "Voin siirtää asian asiakaspalvelumme ihmiselle yhdessä yhteenvedon kanssa.",
        "sv": "Jag kan lämna över ärendet till vår kundtjänst med en sammanfattning."},
    "security_specialist": {
        "en": "A security specialist from our team can guide you through cleaning the device, and I will pass on a summary.",
        "fi": "Tietoturva-asiantuntijamme voi opastaa laitteen puhdistamisessa, ja välitän hänelle yhteenvedon.",
        "sv": "En säkerhetsexpert hos oss kan guida dig, och jag skickar vidare en sammanfattning."},
    "business_security_support": {
        "en": "Our business security support can help your company step by step, and I will pass on a summary of what we know.",
        "fi": "Yritysasiakkaiden tietoturvatukemme voi auttaa yritystänne vaihe vaiheelta, ja välitän heille yhteenvedon.",
        "sv": "Vårt säkerhetsstöd för företag kan hjälpa er steg för steg, och jag skickar vidare en sammanfattning."},
    "human_advisor": {
        "en": "A trained advisor from our team can help you plan the next steps safely, and I will pass on a summary.",
        "fi": "Koulutettu neuvojamme voi auttaa sinua suunnittelemaan seuraavat askeleet turvallisesti, ja välitän hänelle yhteenvedon.",
        "sv": "En utbildad rådgivare hos oss kan hjälpa dig planera nästa steg på ett säkert sätt."},
}

# "Safe branch" conditional for twins whose benign answer needs no action.
SAFE_COND = {
    "self_initiated": {"en": "If you made this change yourself then this is a normal situation and nothing else needs to be done.",
                       "fi": "Jos teit muutoksen itse niin tämä on tavallinen tilanne eikä muuta tarvitse tehdä.",
                       "sv": "Om du gjorde ändringen själv så är allt normalt och du behöver inte göra något."},
    "_default": {"en": "If none of this happened then this is a normal situation and nothing else needs to be done.",
                 "fi": "Jos mitään tästä ei tapahtunut niin tämä on tavallinen tilanne eikä muuta tarvitse tehdä.",
                 "sv": "Om inget av detta hände så är allt normalt och du behöver inte göra något."},
}

# Conditional prefixes for twin single-turn answers ("safe if the riskier answer
# is true"). Written WITHOUT a comma so the whole sentence stays one
# conditional clause (the unsafe-statement check skips conditional clauses).
COND_PREFIX = {
    "card_details_entered": {"en": "If you typed your card details on the page then", "fi": "Jos laitoit korttitietosi sivulle niin", "sv": "Om du lämnade dina kortuppgifter på sidan så"},
    "code_shared": {"en": "If you gave the caller a code then", "fi": "Jos kerroit soittajalle koodin niin", "sv": "Om du gav uppringaren en kod så"},
    "credentials_entered": {"en": "If you logged in to your bank or typed a password during this then", "fi": "Jos kirjauduit verkkopankkiin tai syötit salasanan tämän aikana niin", "sv": "Om du loggade in på banken eller skrev ett lösenord under tiden så"},
    "self_initiated": {"en": "If you did not make this change yourself then", "fi": "Jos et tehnyt tätä muutosta itse niin", "sv": "Om du inte gjorde ändringen själv så"},
    "payment_status": {"en": "If the payment has already been made then", "fi": "Jos maksu on jo maksettu niin", "sv": "Om betalningen redan är gjord så"},
    "money_transferred": {"en": "If money has already been sent then", "fi": "Jos rahaa on jo siirretty niin", "sv": "Om pengar redan har skickats så"},
    "app_installed": {"en": "If you opened the document and allowed its content then", "fi": "Jos avasit asiakirjan ja sallit sen sisällön niin", "sv": "Om du öppnade dokumentet och tillät innehållet så"},
    "remote_access_given": {"en": "If someone controlled your device remotely then", "fi": "Jos joku ohjasi laitettasi etäyhteydellä niin", "sv": "Om någon styrde din enhet på distans så"},
    "user_safety": {"en": "If you feel unsafe right now then", "fi": "Jos et tunne oloasi turvalliseksi nyt niin", "sv": "Om du inte känner dig trygg nu så"},
}

# ---------------------------------------------------------------------------
# 2. Questions (one topic each; checked against question_topics.json).
# ---------------------------------------------------------------------------
QUESTION_TEXT = {
    "credentials_entered": {"en": "Did you log in to your online bank or type any password while this was going on?",
                            "fi": "Kirjauduitko verkkopankkiin tai syötitkö jonkin salasanan tämän aikana?",
                            "sv": "Loggade du in på banken eller skrev du något lösenord under tiden?"},
    "card_details_entered": {"en": "Did you type your card number on the page?",
                             "fi": "Laitoitko sivulle korttisi tiedot?",
                             "sv": "Lämnade du ditt kortnummer på sidan?"},
    "code_shared": {"en": "Did you give anyone the code you received?",
                    "fi": "Kerroitko kenellekään saamasi koodin?",
                    "sv": "Gav du någon koden som du fick?"},
    "money_transferred": {"en": "Has any money already been moved?",
                          "fi": "Ehditkö jo siirtää rahaa?",
                          "sv": "Har du redan skickat några pengar?"},
    "app_installed": {"en": "Did you open the file and allow its content, or install anything?",
                      "fi": "Avasitko tiedoston ja sallitko sen sisällön, tai asensitko jotain?",
                      "sv": "Öppnade du filen eller installerade du något?"},
    "remote_access_given": {"en": "Did anyone control your device remotely?",
                            "fi": "Pääsikö joku ohjaamaan laitettasi etäyhteydellä?",
                            "sv": "Fick någon styra din enhet med fjärrstyrning?"},
    "link_details": {"en": "Could you share the web address of the link?",
                     "fi": "Voitko kertoa linkin osoitteen?",
                     "sv": "Kan du berätta länkens adress?"},
    "call_details": {"en": "What did the caller ask you to do?",
                     "fi": "Mitä soittaja pyysi sinua tekemään?",
                     "sv": "Vad sa de i samtalet?"},
    "self_initiated": {"en": "Did you make this change or order yourself?",
                       "fi": "Teitkö tämän muutoksen tai tilauksen itse?",
                       "sv": "Gjorde du den här ändringen själv?"},
    "device_behaviour": {"en": "Is the phone behaving strangely, for example with pop-ups or fast battery drain?",
                         "fi": "Toimiiko puhelin oudosti, esimerkiksi ponnahdusikkunoiden tai akun kanssa?",
                         "sv": "Beter sig telefonen konstigt?"},
    "signal_lost": {"en": "Does your phone still have network service?",
                    "fi": "Onko puhelimessasi vielä kenttä?",
                    "sv": "Har telefonen fortfarande täckning?"},
    "account_affected": {"en": "Which account or service is this about?",
                         "fi": "Mitä palvelua tai tiliä tämä koskee?",
                         "sv": "Vilket konto eller vilken tjänst gäller det?"},
    "password_reused": {"en": "Do you use the same password on other services?",
                        "fi": "Käytätkö samaa salasanaa muissa palveluissa?",
                        "sv": "Använder du samma lösenord i andra tjänster?"},
    "business_access": {"en": "Who in your company has admin rights or approves payments?",
                        "fi": "Kenellä yrityksessänne on ylläpitäjän oikeudet tai maksujen hyväksyntä?",
                        "sv": "Vilka i företaget har admin-behörighet eller godkänner betalningar?"},
    "payment_status": {"en": "Has the payment already been made?",
                       "fi": "Onko maksu jo maksettu?",
                       "sv": "Är betalningen redan gjord?"},
    "user_safety": {"en": "Are you safe right now, and could someone close to you have access to your phone?",
                    "fi": "Oletko nyt turvassa, ja voiko joku läheisesi päästä käsiksi puhelimeesi?",
                    "sv": "Är du trygg just nu?"},
    "time_of_event": {"en": "When did this happen?", "fi": "Milloin tämä tapahtui?", "sv": "När hände det?"},
    "message_content": {"en": "What does the message say exactly?", "fi": "Mitä viestissä tarkalleen lukee?", "sv": "Vad står det i meddelandet?"},
}

# Context-specific variants (chosen by concept features) — same topic, clearer wording.
QUESTION_VARIANTS = {
    ("self_initiated", "sim"): {"en": "Did you order the new SIM or eSIM change yourself?",
                                "fi": "Tilasitko uuden SIMin tai eSIM-muutoksen itse?",
                                "sv": "Beställde du det nya SIM-kortet eller eSIM-ändringen själv?"},
    ("self_initiated", "login_alert"): {"en": "Was that sign-in made by you yourself, for example while travelling?",
                                        "fi": "Olitko se itse sinä, esimerkiksi matkoilla?",
                                        "sv": "Var det du själv, till exempel på resa?"},
    ("self_initiated", "order"): {"en": "Did you or a colleague place this order yourselves?",
                                  "fi": "Tilasitko sinä tai joku kollegasi nämä itse?",
                                  "sv": "Beställde du eller en kollega detta själv?"},
    ("code_shared", "voice_call"): {"en": "Did you give the code to the person you spoke with?",
                                    "fi": "Kerroitko koodin henkilölle, jonka kanssa puhuit?",
                                    "sv": "Gav du koden till personen du pratade med?"},
}

CONSENT_WHAT = {
    "subscription.recent_events": {"en": "recent SIM, eSIM, number and order events on your subscription",
                                   "fi": "liittymäsi viimeaikaiset SIM-, eSIM-, numero- ja tilaustapahtumat",
                                   "sv": "senaste SIM-, eSIM-, nummer- och orderhändelser för ditt abonnemang"},
    "subscription.network_status": {"en": "whether your line has network service right now",
                                    "fi": "onko liittymässäsi nyt verkkoyhteys",
                                    "sv": "om ditt abonnemang har täckning just nu"},
    "device.recent_app_installs": {"en": "apps installed on this device in the last two weeks",
                                   "fi": "tähän laitteeseen kahden viime viikon aikana asennetut sovellukset",
                                   "sv": "appar som installerats på enheten de senaste två veckorna"},
    "device.security_findings": {"en": "what the security app found on this device, such as remote control apps",
                                 "fi": "mitä tietoturvasovellus on löytänyt laitteesta, esimerkiksi etäohjaussovellukset",
                                 "sv": "vad säkerhetsappen hittat på enheten, till exempel fjärrstyrningsappar"},
    "operator_account.recent_logins": {"en": "recent logins to your account with us",
                                       "fi": "viimeaikaiset kirjautumiset tiliisi meillä",
                                       "sv": "senaste inloggningar på ditt konto hos oss"},
    "credential_exposure": {"en": "the breach monitoring result for your email address",
                            "fi": "sähköpostiosoitteesi tietovuotoseurannan tuloksen",
                            "sv": "resultatet av läckagekontrollen för din e-postadress"},
    "business.staff_accounts": {"en": "your company's user list and when each user was last active",
                                "fi": "yrityksenne käyttäjälistan ja kunkin viimeisimmän aktiivisuuden",
                                "sv": "företagets användarlista och när varje användare senast var aktiv"},
}
CONSENT_FRAME = {
    "en": "To give you more precise advice, may I look at {what}? I will only use it for this question, and I can still help if you say no.",
    "fi": "Jotta voin neuvoa tarkemmin, saanko katsoa {what}? Käytän tietoa vain tähän asiaan, ja autan myös, jos et halua.",
    "sv": "För att ge mer exakta råd, får jag titta på {what}? Jag använder det bara för den här frågan, och jag hjälper dig även om du säger nej.",
}
AND_WORD = {"en": " and ", "fi": " sekä ", "sv": " och "}

ACK = {  # conversational glue for the app (not part of advice_text)
    "thanks": {"en": "Thanks, that helps.", "fi": "Kiitos, tämä auttaa.", "sv": "Tack, det hjälper."},
    "consent_yes": {"en": "Thank you, I checked that information.", "fi": "Kiitos, tarkistin tiedot.", "sv": "Tack, jag kontrollerade uppgifterna."},
    "consent_no": {"en": "No problem, I will advise without it.", "fi": "Selvä, neuvon ilman niitä tietoja.", "sv": "Inga problem, jag ger råd utan det."},
    "unsure": {"en": "That's fine. I'll assume the safer case to be careful.", "fi": "Ei haittaa. Oletan varmuuden vuoksi riskialttiimman vaihtoehdon.", "sv": "Det gör inget. Jag utgår från det säkrare alternativet."},
    "ask_more": {"en": "Could you tell me a bit more about what happened, or paste the message you received?",
                 "fi": "Voisitko kertoa hieman tarkemmin, mitä tapahtui, tai liittää saamasi viestin?",
                 "sv": "Kan du berätta lite mer om vad som hände, eller klistra in meddelandet du fick?"},
}

# ---------------------------------------------------------------------------
# 3. Concept lexicon (fi / en / sv). These are the generalising features: they
#    describe WHAT is going on, independent of brand names and wording.
# ---------------------------------------------------------------------------
CONCEPTS = {
    "pay_request": r"\b(pay|payment|fee)\b|maksa|maksu|betala|avgift|€|\beur\b|euro",
    "card": r"\bcard|kortti|kortin|korttitied|\bkort",
    "code": r"\bcodes?\b|koodi|\bkod(en|er)?\b|\bpin\b|tunnuslu|vahvistusnum",
    "password": r"password|salasan|lösenord",
    "bank_login": r"verkkopankki|online bank|bankkod|bank ?id|identification|identif|tunnistau|pankkitunnu|identifiering|credentials",
    "bank": r"\bbank|pankki|pankin|banken",
    "link": r"https?://|\blink|linkki|linkin|länk",
    "qr": r"\bqr",
    "installed": r"\b(installed|downloaded)\b|asensin|asennettu|latasin|installerade|laddade ner",
    "install_req": r"\binstall|asenna|asentamalla|installera",
    "app_profile": r"\bapp\b|sovellu|profiil|profile|certificate|varmenne|certifikat|\bapk\b",
    "remote": r"remote|etätuk|etäyhte|etäohjel|anydesk|teamviewer|controlled my|ohjasi|fjärr",
    "sim": r"\besim|\bsim\b|sim-|liittymä|abonnemang|\bsubscriptions?\b",
    "porting": r"(number|numero|nummer)[^.]{0,40}(moved|transfer|siirt|flytt|port)|siirtopyyntö|porting|toiselle operaattorille|another operator",
    "no_service": r"no service|lost (signal|service)|kenttä (katosi|hävisi|lähti)|ei kenttää|täckning|lost signal|no_service",
    "login_alert": r"(signed|logged) in from|new device|uudelta laitteelta|kirjauduttiin|ny enhet|inloggning från|espanja|spain|spanien",
    "approval_prompt": r"approv|hyväksy|godkänn",
    "repeated": r"repeated|\d+ (kertaa|times|gånger)|keeps|jatkuvasti|toistuvasti|hela tiden|again and again",
    "breach": r"breach|leak|tietomurto|vuot|läck|dataintrång|data leak",
    "only_email": r"only (my )?email|names and email|nimet ja sähköposti|vain sähköposti|bara e-post|endast e-post|email addresses were exposed|names and e-?mail",
    "hashed_pw": r"hashed|salatut salasanat|hashade|passwords were|salasanat",
    "invoice": r"invoice|lasku|faktura",
    "bank_change": r"(new|uusi|uudelle|nytt|nya)[^.]{0,20}(bank account|account number|tilinumero|tili\b|kontonummer|konto)|bank (has )?chang|pankki vaihtuu|tilinumero|account number",
    "verified_by_phone": r"(called|soitin|ringde)[^.]{0,80}(confirm|vahvist|bekräft)|confirmed the change|vahvisti muutoksen|bekräftade",
    "ceo": r"\bceo\b|toimitusjohtaj|\bvd\b|managing director|\bboss\b|pomo",
    "confidential": r"confidential|luottamuksellin|konfidentiell|keep it (quiet|secret)|in a meeting|palaverissa|kokouksessa",
    "urgency": r"urgent|immediately|within \d+|kiire|heti|tänään|\d+ ?(h|tunnin|hours|timmar)\b|brådskande|idag|today",
    "attachment": r"attach|\.zip|\.pdf|\.doc|liite|bilaga|\bzip\b",
    "enabled_content": r"enable content|enable editing|salli(n)? (sisäl|muokk)|aktiverade innehåll|clicked enable",
    "ransom": r"files? (are|were|have been) encrypted|encrypted (files|my files)|tiedosto\w* (on |ovat )?salattu|salattiin|ransom|lunnas|lunnai|bitcoin|filerna är krypterade|krypterats|lösensumma",
    "ex_partner": r"\bex\b|\bex-|former partner|entinen (kumppani|puoliso|mies|vaimo|poikaystävä|tyttöystävä)|exä|expartner|före detta",
    "monitoring": r"know(s)? (what|where)|seems to know|tietä(ä|vän) (mitä|missä)|tietää missä|vet (vad|var)|tracking|seuranta|stalk|vakoil|övervak|seuraa minua",
    "trace_request": r"huijarin (osoite|nimi|henkilöllisyys|tiedot)|etsi[^.]{0,20}huijari|scammer's (address|name|identity)|bedragarens (adress|namn)|who owns|find out who|kuka omistaa|selvittää kuka|vem som äger|vem äger|ta reda på vem|publish|julkaista|publicera|expose",
    "family": r"\bmum\b|\bmom\b|\bdad\b|äiti|isä|iskä|mamma|pappa|grandchild|lapsenlap|barnbarn|my child|lapseni|mitt barn|new number|uusi numero|nytt nummer",
    "investment": r"invest|sijoit|trading|krypto|crypto|tuotto|avkastning",
    "prize": r"\bwon\b|winner|voitit|voittanut|gift card|lahjakort|presentkort|vunnit|prize|palkinto",
    "parcel": r"parcel|shipment|delivery|courier|customs|paketti|lähetys|toimitus|kuriiri|tulli|\bpaket|leverans|\btull",
    "police": r"police|poliisi|polis",
    "safe_account": r"safe account|turvatili|säkert konto|secure account",
    "did_enter": r"\b(typed|entered|gave|read (out|him|her|them)|filled|logged in|logged into|logged on|signed in|i read)\b|syötin|kirjoitin|annoin|luin|täytin|kirjauduin|skrev in|angav|\bgav\b|läste upp|loggade in",
    "did_pay": r"\b(already (paid|moved|sent)|i paid|we paid|i moved|i transferred)\b|maksoin|maksoimme|siirsin|siirsimme|betalade|förde över|flyttade",
    "did_click": r"\b(clicked|opened|tapped)\b|klikkasin|avasin|painoin|klickade|öppnade",
    "negated": r"didn't|did not|haven't|have not|not (paid|yet)|en ole (vielä |ehtinyt )?(maksanut|syöttänyt|antanut|kirjautunut|avannut|asentanut|tehnyt|tilannut)|inte (betalat|gjort|skrivit)|har inte|hung up|en (syöttänyt|antanut|kirjautunut|maksanut|avannut|asentanut|tehnyt)|suljin puhelun|lopetin puhelun|lade på|before doing anything|ennen kuin tein mitään",
    "own_action": r"myself|i ordered|i booked|we booked|logging in .{0,20}right now|i'?m logging|itse|tilasin|varasimme|kirjaudun (juuri|nyt|parhaillaan)|själv|beställde|bokade|\busual\b|tavallinen|tavanomai|vanlig",
    "no_link_mention": r"no link|ei linkkiä|ingen länk|without (a )?link|ilman linkkiä|there'?s no link",
    "injection": r"verified (as )?safe|tarkistettu turvalliseksi|turvalliseksi|ai assistant|tekoäly|assistant|ignore (previous|all)|verifierad|säker enligt",
    "router": r"router|reititin|reitittim",
    "domain": r"\bdomain|verkkotunnu|domän",
    "marketplace": r"selling|\bsell\b|buyer|myyn|ostaja|säljer|köpare|marketplace|\btori\b|receiving the payment|vastaanottaa maksun",
    "voice_call": r"\bcall|soitti|puhelu|ringde|samtal|called|soitto",
    "missed_call": r"missed call|one ring|rang once|vastaamat|soi kerran|missade samtal|ringde en gång|one-ring",
    "certificate_profile": r"certificate|varmenne|certifikat|configuration profile|hallintaprofiili|konfigurationsprofil",
    "forwarding": r"forward|edelleenläh|vidarebefordr|fake invoices from|huijauslaskuja|receive (fake )?invoices",
    "former_employee": r"(former|left|lähtenyt|entinen) (employee|työntekijä)|employee who left|slutat|före detta anställd|left a month ago|lähti kuukausi",
    "terminal": r"terminal|maksupääte|kassaterminal|till computer|kassakone",
    "order": r"order confirmation|tilausvahvistus|orderbekräftelse|ordered|tilattu|beställ|\border\b|tilaus",
    "phone_strange": r"acts strangely|strange|toimii oudosti|oudosti|konstigt|battery|akku|pop-?up",
    "business": r"company|business|yritys|yrityksen|företag|webshop|verkkokauppa|our customers|asiakkaamme",
    "asks_if_safe": r"is (this|it) (safe|real|trustworthy|genuine|legit)|onko (tämä|se) (aito|turvallinen|luotettava|huijaus)|är (det|detta) (äkta|säkert)",
    "expected": r"i (did )?order(ed)?|i'm expecting|odotan|tilasin|väntar|beställde|i booked|just now|juuri äsken|nyss",
    "no_request": r"asked (for )?nothing|asked me nothing|did not ask|didn't ask|even though i didn't ask|ei pyytänyt mitään|eivät (kysyneet|pyytäneet) mitään|ei kysynyt mitään|vaikka en pyytänyt|bad inte om|frågade inget|asked no",
    "known_number": r"number (in|from) (our|the) contract|number on (my|the) card|kortin takana|known number|tunnettuun numeroon|sopimuksessa",
}
CONCEPT_RX = {k: re.compile(v, re.I | re.U) for k, v in CONCEPTS.items()}


def concepts_in(text):
    t = (text or "").lower()
    return sorted(k for k, rx in CONCEPT_RX.items() if rx.search(t))


# ---------------------------------------------------------------------------
# 4. Reading the user's answers (yes / no / unsure), per topic polarity.
# ---------------------------------------------------------------------------
YES_RX = re.compile(r"^\s*(yes|yeah|yep|yup|sure|i did|i have|kyllä|joo|juu|kyl|ja\b|japp|jo\b)|\b(i did|i have|already|i typed|i entered|i gave|i read|i paid|i installed|i logged|i clicked|tein sen|annoin|luin|syötin|maksoin|asensin|kirjauduin|siirsin|gjorde|gav|läste|betalade|installerade|loggade in)\b", re.I)
NO_RX = re.compile(r"^\s*(no|nope|nah|not|never|ei\b|en\b|emme|nej|inte|aldrig)|\b(didn't|did not|haven't|have not|nothing|never|en (ole|antanut|syöttänyt|kirjautunut|maksanut|avannut|asentanut|tehnyt|lukenut)|ei mitään|ingenting|inte|aldrig|hung up)\b", re.I)
UNSURE_RX = re.compile(r"not sure|don't know|do not know|no idea|can't remember|maybe|en (ole varma|tiedä|muista)|ehkä|vet inte|osäker|inte säker|kanske|minns inte", re.I)
# For these topics a "yes" answer means the SAFER branch.
YES_IS_SAFE = {"self_initiated", "user_safety"}

# ---------------------------------------------------------------------------
# 5. Playbooks for guidance topics that have no dev cases (test-only families).
#    Derived from the guidance passage text in data/reference/guidance_corpus.jsonl.
#    Learned playbooks for the other guidance ids are built in train.py.
# ---------------------------------------------------------------------------
MANUAL_PLAYBOOKS = {
    "guidance_marketplace_scam": {"action": "do_not_interact_and_delete", "risk": "medium", "urgency": "routine", "esc": None,
                                  "steps": ["dont_open_link", "dont_share_codes", "delete_or_report_message"]},
    "guidance_qr_codes": {"action": "verify_via_official_channel", "risk": "medium", "urgency": "routine", "esc": None,
                          "steps": ["dont_pay_or_transfer", "verify_official_channel"]},
    "guidance_config_profile": {"action": "isolate_device_and_scan", "risk": "high", "urgency": "today", "esc": "security_specialist",
                                "steps": ["remove_app_or_profile", "avoid_banking_on_device", "change_password", "use_trusted_device"]},
    "guidance_voice_clone": {"action": "verify_via_official_channel", "risk": "high", "urgency": "immediate", "esc": None,
                             "steps": ["hang_up_dont_call_back", "verify_out_of_band", "dont_pay_or_transfer"]},
    "guidance_wangiri": {"action": "do_not_interact_and_delete", "risk": "low", "urgency": "routine", "esc": None,
                         "steps": ["hang_up_dont_call_back"]},
    "guidance_number_porting": {"action": "contact_operator_support", "risk": "critical", "urgency": "immediate", "esc": "operator_support_agent",
                                "steps": ["contact_operator", "lock_subscription", "dont_reply"]},
    "guidance_router_security": {"action": "change_password_and_enable_mfa", "risk": "medium", "urgency": "today", "esc": None,
                                 "steps": ["secure_router", "update_device", "revoke_access"]},
    "guidance_fake_domain_invoice": {"action": "verify_via_official_channel", "risk": "medium", "urgency": "routine", "esc": None,
                                     "steps": ["dont_pay_or_transfer", "verify_official_channel"]},
}

# Distinctive trigger concepts for the manual playbooks: the playbook is only
# applied when these concepts are present (guards against loose retrieval).
PLAYBOOK_TRIGGERS = {
    "guidance_marketplace_scam": ["marketplace"],
    "guidance_qr_codes": ["qr"],
    "guidance_config_profile": ["certificate_profile"],
    "guidance_voice_clone": ["family", "voice_call"],
    "guidance_wangiri": ["missed_call"],
    "guidance_number_porting": ["porting"],
    "guidance_router_security": ["router"],
    "guidance_fake_domain_invoice": ["domain", "invoice"],
}

URGENCY_FOR_RISK = {"none": "none", "low": "routine", "medium": "routine", "high": "today", "critical": "immediate"}
