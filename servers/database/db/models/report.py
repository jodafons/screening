

class Report(Base):

    __tablename___ = "image_info"
    id             = Column(Integer, primary_key = True)
    version        = Column(Integer, default=0)
    imageinfo_id   = Column(String, unique=True)
    image_id       = Column(String)
    has_tb         = Column(Boolean)
    gender         = Column(Char)
    age            = Column(Integer)
    date_exam      = Column(DateTime)
    date_insertion = Column(DateTime)

    under_penetrated                = Column(Boolean)
    over_penetrated                 = Column(Boolean)
    costophrenic_cropped            = Column(Boolean)
    apices_cropped                  = Column(Boolean)
    reliable_radiography            = Column(Boolean)
    minimum_interpretation_quality  = Column(Boolean)
    performed_by                    = Column(String)
    comment                         = Column(String)
    metadata                        = Column(String, default="{}")

