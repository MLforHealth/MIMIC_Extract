# Original source taken from https://github.com/wboag/mimic-tokenize/blob/master/heuristic-tokenize.py at
# commit e953d271bbb4c53aee5cc9a7b8be870a6b007604

import re, nltk

def is_inline_title(text):
    m = re.search('^([a-zA-Z ]+:) ', text)
    if not m: return False
    return is_title(m.groups()[0])

stopwords = set(['of', 'on', 'or'])
def is_title(text):
    if not text.endswith(':'): return False
    text = text[:-1]

    # be a little loose here... can tighten if it causes errors
    text = re.sub('(\([^\)]*?\))', '', text)

    # Are all non-stopwords capitalized?
    for word in text.split():
        if word in stopwords: continue
        if not word[0].isupper(): return False

    # I noticed this is a common issue (non-title aapears at beginning of line)
    if text == 'Disp': return False

    # optionally: could assert that it is less than 6 tokens
    return True


def sent_tokenize_rules(text):

    # long sections are OBVIOUSLY different sentences
    text = re.sub('---+', '\n\n-----\n\n', text)
    text = re.sub('___+', '\n\n_____\n\n', text)
    text = re.sub('\n\n+', '\n\n', text)

    segments = text.split('\n\n')

    # strategy: break down segments and chip away structure until just prose.
    #           once you have prose, use nltk.sent_tokenize()

    ### Separate section headers ###
    new_segments = []

    # deal with this one edge case (multiple headers per line) up front
    m1 = re.match('(Admission Date:) (.*) (Discharge Date:) (.*)', segments[0])
    if m1:
        new_segments += list(map(lambda s: s.strip(), m1.groups()))
        segments = segments[1:]

    m2 = re.match('(Date of Birth:) (.*) (Sex:) (.*)'            , segments[0])
    if m2:
        new_segments += list(map(lambda s: s.strip(), m2.groups()))
        segments = segments[1:]

    for segment in segments:
        # find all section headers
        possible_headers  = re.findall('\n([A-Z][^\n:]+:)', '\n'+segment)
        #assert len(possible_headers) < 2, str(possible_headers)
        headers = []
        for h in possible_headers:
            #print 'cand=[%s]' % h
            if is_title(h.strip()):
                #print '\tYES=[%s]' % h
                headers.append(h.strip())

        # split text into new segments, delimiting on these headers
        for h in headers:
            h = h.strip()

            # split this segment into 3 smaller segments
            ind = segment.index(h)
            prefix = segment[:ind].strip()
            rest   = segment[ ind+len(h):].strip()

            # add the prefix (potentially empty)
            if len(prefix) > 0:
                new_segments.append(prefix.strip())

            # add the header
            new_segments.append(h)

            # remove the prefix from processing (very unlikely to be empty)
            segment = rest.strip()

        # add the final piece (aka what comes after all headers are processed)
        if len(segment) > 0:
            new_segments.append(segment.strip())

    # copy over the new list of segments (further segmented than original segments)
    segments = list(new_segments)
    new_segments = []


    ### Low-hanging fruit: "_____" is a delimiter
    for segment in segments:
        subsections = segment.split('\n_____\n')
        new_segments.append(subsections[0])
        for ss in subsections[1:]:
            new_segments.append('_____')
            new_segments.append(ss)

    segments = list(new_segments)
    new_segments = []


    ### Low-hanging fruit: "-----" is a delimiter
    for segment in segments:
        subsections = segment.split('\n-----\n')
        new_segments.append(subsections[0])
        for ss in subsections[1:]:
            new_segments.append('-----')
            new_segments.append(ss)

    segments = list(new_segments)
    new_segments = []

    '''
    for segment in segments:
        print '------------START------------'
        print segment
        print '-------------END-------------'
        print
    exit()
    '''

    ### Separate enumerated lists ###
    for segment in segments:
        old_len = len(new_segments)
        if not re.search('\n\s*\d+\.', '\n'+segment): 
            new_segments.append(segment)
            continue

        #print '------------START------------'
        #print segment
        #print '-------------END-------------'
        #print

        # generalizes in case the list STARTS this section
        segment = '\n'+segment

        # determine whether this segment contains a bulleted list (assumes i,i+1,...,n)
        start = int(re.search('\n\s*(\d+)\.', segment).groups()[0])
        n = start
        while re.search('\n\s*%d\.'%n,segment):
            n += 1
        n -= 1

        # no bulleted list
        if n < 1 or (n - start) == 0:
            new_segments.append(segment)
            continue

        #print '------------START------------'
        #print segment
        #print '-------------END-------------'
        #print start,n
        #print 

        # break each list into its own line
        # challenge: not clear how to tell when the list ends if more text happens next
        for i in range(start,n+1):
            matching_text = re.search('(\n\s*\d+\.)',segment).groups()[0]
            prefix  = segment[:segment.index(matching_text) ].strip()
            segment = segment[ segment.index(matching_text):].strip()

            if len(prefix)>0:
                new_segments.append(prefix)

        if len(segment)>0:
            new_segments.append(segment)


        #print 'Out Segments:'
        #for out_segment in new_segments[old_len:]:
        #    print '------------START------------'
        #    print out_segment
        #    print '-------------END-------------'
        #print('\n\n')

    segments = list(new_segments)
    new_segments = []

    '''
        TODO: Big Challenge
        There is so much variation in what makes a list. Intuitively, I can tell it's a
        list because it shows repeated structure (often following a header)
        Examples of some lists (with numbers & symptoms changed around to noise)
            Past Medical History:
            -- Hyperlipidemia
            -- lactose intolerance
            -- Hypertension
            Physical Exam:
            Vitals - T 82.2 BP 123/23 HR 73 R 21 75% on 2L NC
            General - well appearing male, sitting up in chair in NAD
            Neck - supple, JVP elevated to angle of jaw 
            CV - distant heart sounds, RRR, faint __PHI_43__ murmur at
            Labs:
            __PHI_10__ 12:00PM BLOOD WBC-8.8 RBC-8.88* Hgb-88.8* Hct-88.8*
            MCV-88 MCH-88.8 MCHC-88.8 RDW-88.8* Plt Ct-888
            __PHI_14__ 04:54AM BLOOD WBC-8.8 RBC-8.88* Hgb-88.8* Hct-88.8*
            MCV-88 MCH-88.8 MCHC-88.8 RDW-88.8* Plt Ct-888
            __PHI_23__ 03:33AM BLOOD WBC-8.8 RBC-8.88* Hgb-88.8* Hct-88.8*
            MCV-88 MCH-88.8 MCHC-88.8 RDW-88.8* Plt Ct-888
            __PHI_109__ 03:06AM BLOOD WBC-8.8 RBC-8.88* Hgb-88.8* Hct-88.8*
            MCV-88 MCH-88.8 MCHC-88.8 RDW-88.8* Plt Ct-888
            __PHI_1__ 05:09AM BLOOD WBC-8.8 RBC-8.88* Hgb-88.8* Hct-88.8*
            MCV-88 MCH-88.8 MCHC-88.8 RDW-88.8* Plt Ct-888
            __PHI_26__ 04:53AM BLOOD WBC-8.8 RBC-8.88* Hgb-88.8* Hct-88.8*
            MCV-88 MCH-88.8 MCHC-88.8 RDW-88.8* Plt Ct-888
            __PHI_301__ 05:30AM BLOOD WBC-8.8 RBC-8.88* Hgb-88.8* Hct-88.8*
            MCV-88 MCH-88.8 MCHC-88.8 RDW-88.8* Plt Ct-888
            Medications on Admission:
            Allopurinol 100 mg DAILY
            Aspirin 250 mg DAILY
            Atorvastatin 10 mg DAILY
            Glimepiride 1 mg once a week.
            Hexavitamin DAILY
            Lasix 50mg M-W-F; 60mg T-Th-Sat-Sun
            Metoprolol 12.5mg TID
            Prilosec OTC 20 mg once a day
            Verapamil 120 mg SR DAILY
    '''

    ### Remove lines with inline titles from larger segments (clearly nonprose)
    for segment in segments:
        '''
        With: __PHI_6__, MD __PHI_5__
        Building: De __PHI_45__ Building (__PHI_32__ Complex) __PHI_87__
        Campus: WEST
        '''

        lines = segment.split('\n')

        buf = []
        for line in lines:
            if is_inline_title(line):
                if len(buf) > 0: new_segments.append('\n'.join(buf))
                buf = []
            buf.append(line)
        if len(buf) > 0:
            new_segments.append('\n'.join(buf))

    segments = list(new_segments)
    new_segments = []

    # Going to put one-liner answers with their sections 
    # (aka A A' B B' C D D' -->  AA' BB' C DD' )
    N = len(segments)
    for i in range(N):
        # avoid segfaults
        if i==0:
            new_segments.append(segments[i])
            continue
        if segments[i].count('\n') == 0 and is_title(segments[i-1]) and not is_title(segments[i]):
            if (i == N-1) or is_title(segments[i+1]):
                new_segments = new_segments[:-1]
                new_segments.append(segments[i-1] + ' ' + segments[i])
            else: new_segments.append(segments[i])
        else:
            new_segments.append(segments[i])

    segments = list(new_segments)
    new_segments = []

    '''
        Should do some kind of regex to find "TEST: value" in segments?
            Indication: Source of embolism.
            BP (mm Hg): 145/89
            HR (bpm): 80
        Note: I made a temporary hack that fixes this particular problem. 
              We'll see how it shakes out
    '''


    '''
        Separate ALL CAPS lines (Warning... is there ever prose that can be all caps?)
    '''



    '''
    for segment in segments:
        print '------------START------------'
        print segment
        print '-------------END-------------'
        print
    exit()
    '''

    return segments
