#======prompt design======

def build_io_prompt(text):
    """
    IO Prompt: Given only a conversation text, let the model directly output the corresponding surgical stage.
    """
    prompt = f"""Du bist ein KI-Assistent, der mit chirurgischen Eingriffen vertraut ist.  
Die Ärzte führen gerade die perkutane Einführung eines zentralvenösen Katheters durch.  
Hier ist ein Gespräch im Operationssaal：
{text}

Deine Aufgabe:
1. Lies das Gespräch aufmerksam.
2. Bestimme genau, zu welcher der folgenden chirurgischen Phasen dieses Gespräch gehört:
   0-preparation
   1-puncture
   2-GuideWire
   3-CathPlacement
   4-CathPositioning
   5-CathAdjustment
   6-CathControl
   7-Closing
   8-Transition
3. Antworte ausschließlich mit einer dieser Phasen (entweder Nummer oder Name), ohne zusätzliche Erklärungen.
"""
    return prompt


def build_cot_prompt(text):
    """
    CoT Prompt: Explicitly require the model to first give a line of reasoning and then summarize.
    """
    prompt = f"""Du bist ein KI-Assistent, der mit chirurgischen Eingriffen vertraut ist.  
Die Ärzte führen gerade die perkutane Einführung eines zentralvenösen Katheters durch.  
Hier ist ein Gespräch im Operationssaal：
{text}

Bitte befolgen Sie die folgenden Schritte, um zu antworten：
1. Extrahieren Sie Anhaltspunkte aus dem Gespräch – zum Beispiel Schlüsselhandlungen oder den Einsatz von medizinischen Geräten.
2. Ziehen Sie daraus Ihre Schlüsse, um zu bestimmen, in welcher der folgenden chirurgischen Phasen sich das Gespräch höchstwahrscheinlich befindet:
(0-preparation
1-puncture
2-GuideWire
3-CathPlacement
4-CathPositioning
5-CathAdjustment
6-CathControl
7-Closing
8-Transition).
3. Geben Sie Ihren Gedankengang an und erläutern Sie kurz, warum Sie zu dieser Schlussfolgerung kommen(maximal 100 Wörter).
"""
    return prompt


def build_tot_prompt(text):
    """
    ToT Prompt: Guide the model to think from multiple perspectives (Tree-of-Thought).
    """
    prompt = f"""(Du bist ein KI-Assistent, der mit chirurgischen Eingriffen vertraut ist.
Ärztinnen und Ärzte führen derzeit die perkutane Einführung eines zentralvenösen Katheters durch.
Bitte leite aus dem folgenden Dialog die entsprechenden chirurgischen Schritte aus mehreren Perspektiven ab(ähnlich einem „Tree-of-Thought“)：)
{text}

Bitte analysieren Sie dieses Gespräch so, als würden Sie mehrere Gedankenzweige (ähnlich einem „Tree of Thought“) verfolgen.
Betrachten Sie dabei, ob und wie jeder Teil des Gesprächs auf die folgenden Phasen hinweist:

Angle A: Stage 0 - Preparation  
- Wird über anfängliche Vorbereitungen (steriles Umfeld, Patientenlagerung, Anlegen einer Lokalanästhesie usw.) gesprochen?

Angle B: Stage 1 - Puncture  
- Gibt es Hinweise auf das Verwenden einer Nadel zum Punktieren der Zielvene, das Bestätigen des Einstichs oder das Abwarten von Blutrückfluss?

Angle C: Stage 2 - GuideWire  
- Wird das Einführen oder Manipulieren eines Führungsdrahts (Guide Wire) nach der Punktion erwähnt?

Angle D: Stage 3 - CathPlacement  
- Zeigt das Gespräch, dass ein Katheter über den Führungsdraht platziert oder vorgeschoben wird?

Angle E: Stage 4 - CathPositioning  
- Wird besprochen, wie der Katheter korrekt in der Vene oder am Zielort positioniert werden soll?

Angle F: Stage 5 - CathAdjustment  
- Wird darüber gesprochen, den Katheter anzupassen, zu rotieren oder teilweise zurückzuziehen, um die optimale Lage zu erreichen?

Angle G: Stage 6 - CathControl  
- Findet eine Spülung, Durchflusskontrolle, Überprüfung der Durchgängigkeit oder Kontrolle auf mögliche Blockaden statt?

Angle H: Stage 7 - Closing  
- Wird über die finalen Abschlussschritte gesprochen, wie das Nähen, Anlegen eines Verbands oder die Absicherung der Austrittsstelle?

Angle I: Stage 8 - Transition  
- Deutet das Gespräch auf den Abschluss des Eingriffs hin, beispielsweise die Übergabe des Patienten oder das Vorbereiten des nächsten Schritts?

Bewerten Sie nach der Betrachtung aller oben genannten Aspekte, welche Phase am besten zu diesem Gespräch passt.
Geben Sie abschließend eine kurze Erklärung, warum diese Phase am wahrscheinlichsten ist.

Ihre Antwort sollte eine der folgenden Phasen sein:

0-preparation
1-puncture
2-GuideWire
3-CathPlacement
4-CathPositioning
5-CathAdjustment
6-CathControl
7-Closing
8-Transition

Bitte skizzieren Sie zuerst kurz Ihr „Tree-of-Thought“-Vorgehen (Zweig-für-Zweig-Argumentation), und geben Sie dann Ihr abschließendes Urteil.Bitte nenne die eine Phase , die am wahrscheinlichsten ist.

"""
    return prompt