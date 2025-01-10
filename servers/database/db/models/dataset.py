Base = declarative_base()


class Fold(Base):
    __tablename__ = "fold"



class Dataset(Base):
    __tablename__ = "dataset"

    id          = Column(Integer, primary_key = True)
    name        = Column(String, unique=True)
    dataset_id  = Column(String, unique=True)
    public      = Column(Boolean, default=False)
    synthetic   = Column(Boolean, default=False)
    last_update = Column(DateTime)
    comment     = Column(String)

    # Foreign  
    images      = relationship("Image", order_by="Image.id", back_populates="dataset")
