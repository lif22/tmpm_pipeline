from os import path
from datetime import datetime
import spacy, gensim
import pandas as pd
from stages.utils.utils import convertDateString
from tqdm import tqdm
from spacy.matcher import Matcher
import statistics

# settings
TOPIC_LABELS = [None, "Initial Application by Candidate",
        "Automatic Reply", "Internal Communication / Clarification of Requirements",
        "Organizational Communication with Candidate for Clarification of Skills or Interview Invitation",
        "Job Offer / Contract Negotiation", "Application Declined"
        ]

class Message:
    def __init__(self, from_, to, subject, content, meta, trainLabel):
        self.from_ = from_
        self.to = to
        self.subject = subject.replace("Re:", "").replace("RE:", "").replace("Fwd:", "").replace("FWD:", "").replace("Aw:", "").replace("AW:", "")
        self.content = content
        self.meta = meta
        self.trainLabel = trainLabel
        self.detectedLabel = ""
        if "Datetime" not in self.meta.keys():
            raise ValueError("This message does not have datetime information which is necessary!")
    
    @staticmethod
    def mapDetectedCategory(id):
        if id == 1:
            return TOPIC_LABELS[1]
        elif id == 2:
            return TOPIC_LABELS[6]
        elif id in (9,6):
            return TOPIC_LABELS[5]
        elif id in (7,0,4):
            return TOPIC_LABELS[4]
        elif id == 3:
            return TOPIC_LABELS[3]
        elif id == 5:
            return TOPIC_LABELS[2]
        elif id == 8: # not mapped
            return "?"

class Case:
    def __init__(self, rootMessage):
        # each case cluster is defined by the ID of its first message (root message)
        self.clusterId = rootMessage.meta["Message-ID"]
        self.rootMessage = rootMessage
        self.messages = [rootMessage]
        self.actors = [rootMessage.from_, rootMessage.to]
    def add(self, message):
        self.messages.append(message)
        self.actors.extend([message.from_, message.to])
        self.actors = list(set(self.actors))
    
    def getCaseDuration(self, timest_format):
        lowest = datetime(1970,1,1)
        highest = datetime(1970,1,1)
        for idx, message in enumerate(self.messages):
            if idx == 0:
                lowest = datetime.strptime(message.meta["Datetime"], timest_format)
                highest = datetime.strptime(message.meta["Datetime"], timest_format)
                continue
            if (datetime.strptime(message.meta["Datetime"], timest_format) > highest):
                highest = datetime.strptime(message.meta["Datetime"], timest_format)
            if (datetime.strptime(message.meta["Datetime"], timest_format) < lowest):
                lowest = datetime.strptime(message.meta["Datetime"], timest_format)
        return lowest, highest, (highest-lowest).total_seconds()
    
    def getMessageCount(self):
        return len(self.messages)
    def getHeadCount(self):
        return len(self.actors)
    def checkCaseSuccess(self):
        for message in self.messages:
            if message.detectedLabel == 2:
                return False
        return True
        
class CasesList(list):
    def findClusterByMessageId(self, id):
        for case in self:
            for message in case.messages:
                if message.meta["Message-ID"] == id:
                    return case.clusterId
    
    def addMessageToCluster(self, clusterId, message):
        for case in self:
            if case.clusterId == clusterId:
                case.add(message)
                
    def prettyprint(self):
        for case in self:
            print("################################################################")
            print(f"Case {case.clusterId}, involved actors: {str(case.actors)}")
            for m in case.messages:
                print("Message:\n---")
                print(f"Header: {m.meta['Datetime']} {m.meta['Message-ID']} {m.meta['In-Reply-To']}")
                print(f"From: {m.from_}")
                print(f"To: {m.to}")
                print(f"Message:\n{m.content}")
    
    def getCorpora(self):
        corpora = []
        for case in self:
            for message in case.messages:
                corpora.append(f"{message.subject} {message.content}") 
        return corpora
    
    def getTrainLabels(self):
        labels = []
        for case in self:
            for message in case.messages:
                if message.trainLabel not in labels:
                    labels.append(message.trainLabel)
        return labels
    
    def getMessagesByLabel(self, label):
        out = []
        for case in self:
            for message in case.messages:
                if message.trainLabel == label:
                    out.append(message)
        return out
    
    def generateEventLog(self, name, filePath):
        log = []
        for case in self:
            for m in case.messages:
                line = dict()
                line["Case"] = case.clusterId
                line["Date"] = m.meta["Datetime"]
                line["Action"] = Message.mapDetectedCategory(m.detectedLabel)
                log.append(line)
                pdlog = pd.DataFrame(log)
                fileName = f"eventlog_{name}.csv"
                pdlog.to_csv(path.join(filePath, fileName), index=False, sep=";")
        return fileName

    def generateDebugLog(self, name, filePath):
        log = []
        for case in self:
            for m in case.messages:
                line = dict()
                line["Date"] = m.meta["Datetime"]
                line["Subject"] = m.subject
                line["Content"] = m.content
                line["DetectedLabel"] = m.detectedLabel
                line["TrainLabel"] = m.trainLabel
                log.append(line)
                pdlog = pd.DataFrame(log)
                fileName = f"debuglog_{name}.csv"
                pdlog.to_csv(path.join(filePath, fileName), index=False, sep=";")
        return fileName
    
    def getMedianCaseDuration(self, timest_format):
        alldurations = []
        for case in self:
            alldurations.append(case.getCaseDuration(timest_format)[2])
        return statistics.median(alldurations)
    
    def getMedianCaseMessageCount(self):
        msgCounts = []
        for case in self:
            msgCounts.append(len(case.messages))
        return statistics.median(msgCounts)
    
    def getMedianCaseHeadcount(self):
        headCount = []
        for case in self:
            headCount.append(len(case.actors))
        return statistics.median(headCount)
            
    @staticmethod
    def groupCases(file, maxDays):
        cases = CasesList()
        # create new column to indicate whether message has been added to case or not
        file["assignedToCase"] = False
        for index, row in tqdm(file.iterrows(), total=file.shape[0]):
            from_ = row["From"]
            to = row["To"]
            subject = row["Subject"]
            content = row["Content"]
            meta = {
                "Datetime": row["Datetime"],
                "Message-ID": row["Message-ID"],
                "In-Reply-To": row["In-Reply-To"]
            }
            train_label = row["Label"]
            m = Message(from_, to, subject, content, meta, train_label)
            if not pd.isnull(meta["In-Reply-To"]):
                # check in-reply-to field to creat thread
                clusterId = cases.findClusterByMessageId(meta["In-Reply-To"])
                if clusterId:
                    # add message to existing cluster
                    cases.addMessageToCluster(clusterId, m)
                    file.loc[index, "assignedToCase"] = True
                else:
                    # create new case
                    case = Case(m)
                    cases.append(case)
                    file.loc[index, "assignedToCase"] = True
            else:
                # no in-reply-to was found, check from/to and datetime to assign messages to cases
                # check all existing cases all messages if from or to matches and if date is not too far away
                msg_dt = convertDateString(m.meta["Datetime"])
                found = False
                for case in cases:
                    if found: break
                    for message in case.messages:
                        if found: break
                        if message.from_ in (m.from_, m.to) or message.to in (m.from_, m.to):
                            curr_msg_dt = convertDateString(message.meta["Datetime"])
                            if abs((curr_msg_dt-msg_dt).days) <= maxDays:
                                cases.addMessageToCluster(case.clusterId, m)
                                file.loc[index, "assignedToCase"] = True
                                found = True
                            else:
                                continue # check other messages in case
                if not found:
                    # if no similarity in sender/receiver was found or if message sending dates are too far apart, create new case from message        
                    c = Case(m)
                    cases.append(c)
                    file.loc[index, "assignedToCase"] = True
        return cases

def lemmatize(corpora, pos_tags=["NOUN", "ADJ", "VERB", "ADV", "PROPN", "DOBJ"]):
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    out = []
    for corpus in corpora:
        doc = nlp(corpus)
        corpus_l = []
        corpus_l.extend([t.lemma_ for t in doc if (t.pos_ in pos_tags and not t.is_stop)])
        out.append(corpus_l)
    return out
    
def gen_words(corpora):
    out = []
    for corpus in corpora:
        corpustext = " ".join(corpus)
        new = gensim.utils.simple_preprocess(corpustext, deacc=True)
        new = [x.replace("want", "") for x in new]
        out.append(new)
    return out