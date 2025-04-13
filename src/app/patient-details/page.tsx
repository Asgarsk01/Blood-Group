"use client"

import type React from "react"

import { useState, useEffect } from "react"
import { useRouter } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Textarea } from "@/components/ui/textarea"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog"
import { toast } from "sonner"
import { Edit, Trash2, ArrowLeft, RefreshCw, FileText, Droplet } from "lucide-react"
import { saveData, getData, STORAGE_KEYS } from "@/lib/utils/storage"

// Mock patient data type
interface Patient {
  id: string
  name: string
  age: number
  gender: string
  email: string
  collectionDate: string
  height: number
  weight: number
  bloodPressure: string
  medicalHistory?: string
  detectionHistory: string
}

export default function PatientDetails() {
  const router = useRouter()
  const [patients, setPatients] = useState<Patient[]>([])
  const [isLoaded, setIsLoaded] = useState(false)
  const [filterValue, setFilterValue] = useState("")
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false)
  const [patientToDelete, setPatientToDelete] = useState<string | null>(null)
  const [refreshKey, setRefreshKey] = useState(0) // Used to force re-render
  const [lastAddedPatient, setLastAddedPatient] = useState<Patient | null>(null) // Track the last added patient

  // Form state
  const [formData, setFormData] = useState({
    id: "",
    name: "",
    age: "",
    gender: "",
    email: "",
    collectionDate: "",
    height: "",
    weight: "",
    bloodPressure: "",
    medicalHistory: "",
  })
  const [isEditing, setIsEditing] = useState(false)

  // Load data
  useEffect(() => {
    // Load data from local storage
    const savedPatients = getData<Patient[]>(STORAGE_KEYS.PATIENTS_LIST, []);

    if (savedPatients.length > 0) {
      setPatients(savedPatients)
      console.log("Loaded patients from storage:", savedPatients)
    } else {
      // If no saved data, use mock data
      loadMockData()
    }

    setIsLoaded(true)
  }, [refreshKey])

  const loadMockData = () => {
    const mockPatients: Patient[] = [
      {
        id: "PAT-2025-0001",
        name: "Jane Smith",
        age: 35,
        gender: "Female",
        email: "jane.smith@gmail.com",
        collectionDate: "2025-03-15",
        height: 165,
        weight: 60,
        bloodPressure: "120/80",
        medicalHistory: "None",
        detectionHistory: "A+",
      },
      {
        id: "PAT-2025-0002",
        name: "John Doe",
        age: 42,
        gender: "Male",
        email: "john.doe@gmail.com",
        collectionDate: "2025-03-14",
        height: 180,
        weight: 85,
        bloodPressure: "130/85",
        medicalHistory: "Hypertension",
        detectionHistory: "O-",
      },
      {
        id: "PAT-2025-0003",
        name: "Alice Johnson",
        age: 28,
        gender: "Female",
        email: "alice.j@gmail.com",
        collectionDate: "2025-03-10",
        height: 170,
        weight: 65,
        bloodPressure: "110/70",
        medicalHistory: "",
        detectionHistory: "Not Tested",
      },
    ]
    setPatients(mockPatients)
    // Save mock data to local storage
    saveData(STORAGE_KEYS.PATIENTS_LIST, mockPatients);
  }

  const generatePatientId = () => {
    const year = new Date().getFullYear()
    const lastId = patients.length > 0 ? Number.parseInt(patients[patients.length - 1].id.split("-")[2]) : 0
    const newId = `PAT-${year}-${String(lastId + 1).padStart(4, "0")}`
    setFormData({ ...formData, id: newId })
  }

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    const { name, value } = e.target
    setFormData({ ...formData, [name]: value })
  }

  const handleSelectChange = (name: string, value: string) => {
    setFormData({ ...formData, [name]: value })
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()

    // Validate required fields
    const requiredFields = ["name", "age", "gender", "email", "collectionDate", "height", "weight", "bloodPressure"]
    for (const field of requiredFields) {
      if (!formData[field as keyof typeof formData]) {
        toast.error("Please fill all required fields");
        return
      }
    }

    // Email validation
    if (!formData.email.endsWith("@gmail.com")) {
      toast.error("Please use a Gmail address");
      return
    }

    let updatedPatients: Patient[] = []
    let savedPatient: Patient

    if (isEditing) {
      // Update existing patient
      savedPatient = {
        id: formData.id,
        name: formData.name,
        age: Number.parseInt(formData.age),
        gender: formData.gender,
        email: formData.email,
        collectionDate: formData.collectionDate,
        height: Number.parseInt(formData.height),
        weight: Number.parseInt(formData.weight),
        bloodPressure: formData.bloodPressure,
        medicalHistory: formData.medicalHistory,
        detectionHistory: patients.find((p) => p.id === formData.id)?.detectionHistory || "Not Tested",
      }

      updatedPatients = patients.map((patient) => (patient.id === formData.id ? savedPatient : patient))
      setPatients(updatedPatients)

      toast.success("Patient updated successfully");
    } else {
      // Generate ID if not already done
      if (!formData.id) {
        const year = new Date().getFullYear()
        const lastId = patients.length > 0 ? Number.parseInt(patients[patients.length - 1].id.split("-")[2]) : 0
        formData.id = `PAT-${year}-${String(lastId + 1).padStart(4, "0")}`
      }

      // Add new patient
      savedPatient = {
        id: formData.id,
        name: formData.name,
        age: Number.parseInt(formData.age),
        gender: formData.gender,
        email: formData.email,
        collectionDate: formData.collectionDate,
        height: Number.parseInt(formData.height),
        weight: Number.parseInt(formData.weight),
        bloodPressure: formData.bloodPressure,
        medicalHistory: formData.medicalHistory,
        detectionHistory: "Not Tested",
      }

      updatedPatients = [...patients, savedPatient]
      setPatients(updatedPatients)

      // Store the last added patient for quick access
      setLastAddedPatient(savedPatient)

      toast.success("Patient added successfully. You can now proceed to blood detection.");
    }

    // Save updated patients to local storage
    saveData(STORAGE_KEYS.PATIENTS_LIST, updatedPatients);

    // Reset form
    resetForm()
  }

  const resetForm = () => {
    setFormData({
      id: "",
      name: "",
      age: "",
      gender: "",
      email: "",
      collectionDate: "",
      height: "",
      weight: "",
      bloodPressure: "",
      medicalHistory: "",
    })
    setIsEditing(false)
  }

  const handleEdit = (patient: Patient) => {
    setFormData({
      id: patient.id,
      name: patient.name,
      age: patient.age.toString(),
      gender: patient.gender,
      email: patient.email,
      collectionDate: patient.collectionDate,
      height: patient.height.toString(),
      weight: patient.weight.toString(),
      bloodPressure: patient.bloodPressure,
      medicalHistory: patient.medicalHistory || "",
    })
    setIsEditing(true)

    // Clear last added patient when editing
    setLastAddedPatient(null)
  }

  const handleDelete = (id: string) => {
    setPatientToDelete(id)
    setDeleteDialogOpen(true)
  }

  const confirmDelete = () => {
    if (patientToDelete) {
      const updatedPatients = patients.filter((patient) => patient.id !== patientToDelete)
      setPatients(updatedPatients)
      // Save updated patients to local storage
      saveData(STORAGE_KEYS.PATIENTS_LIST, updatedPatients);

      // If we deleted the last added patient, clear it
      if (lastAddedPatient && lastAddedPatient.id === patientToDelete) {
        setLastAddedPatient(null)
      }

      toast.success("Patient deleted successfully");
    }
    setDeleteDialogOpen(false)
    setPatientToDelete(null)
  }

  const handleProceedToDetection = (patient: Patient) => {
    // Store selected patient for use in detection screen
    saveData(STORAGE_KEYS.SELECTED_PATIENT, patient);
    console.log("Selected patient for detection:", patient)

    // Also update the patientsList to ensure consistency
    const updatedPatients = patients.map((p) => (p.id === patient.id ? patient : p));
    saveData(STORAGE_KEYS.PATIENTS_LIST, updatedPatients);

    // Navigate to blood detection page
    router.push("/blood-detection")
  }

  // Add a function to handle viewing a patient's report
  const handleViewReport = (patient: Patient) => {
    if (patient.detectionHistory === "Not Tested") {
      toast.error("This patient has not been tested yet. Please perform blood detection first.");
      return
    }

    // Store selected patient for use in report preview
    saveData(STORAGE_KEYS.SELECTED_PATIENT, patient);
    saveData(STORAGE_KEYS.DETECTION_RESULT, patient.detectionHistory);

    // Navigate to report preview
    router.push("/report-preview")
  }

  const handleRefreshData = () => {
    // Force a refresh of the patient data from local storage
    setRefreshKey((prevKey) => prevKey + 1)
    setLastAddedPatient(null) // Clear last added patient on refresh
    toast.success("Patient list has been refreshed");
  }

  const filteredPatients = patients.filter(
    (patient) =>
      patient.id.toLowerCase().includes(filterValue.toLowerCase()) ||
      patient.name.toLowerCase().includes(filterValue.toLowerCase()),
  )

  if (!isLoaded) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <p className="text-lg">Loading patient data...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white shadow">
        <div className="container mx-auto px-4 py-4 flex justify-between items-center">
          <h1 className="text-2xl font-bold">Patient Details</h1>
          <Button variant="outline" onClick={() => router.push("/dashboard")} className="flex items-center gap-2">
            <ArrowLeft className="h-4 w-4" />
            Back to Dashboard
          </Button>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8">
        <Card className="mb-8">
          <CardContent className="pt-6">
            <form onSubmit={handleSubmit} className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                <div className="space-y-2">
                  <Label htmlFor="name">Full Name *</Label>
                  <Input
                    id="name"
                    name="name"
                    value={formData.name}
                    onChange={handleInputChange}
                    placeholder="Full Name"
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="age">Age *</Label>
                  <Input
                    id="age"
                    name="age"
                    type="number"
                    value={formData.age}
                    onChange={handleInputChange}
                    placeholder="Age"
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="gender">Gender *</Label>
                  <Select value={formData.gender} onValueChange={(value) => handleSelectChange("gender", value)}>
                    <SelectTrigger id="gender">
                      <SelectValue placeholder="Select Gender" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="Male">Male</SelectItem>
                      <SelectItem value="Female">Female</SelectItem>
                      <SelectItem value="Other">Other</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="email">Gmail Address *</Label>
                  <Input
                    id="email"
                    name="email"
                    type="email"
                    value={formData.email}
                    onChange={handleInputChange}
                    placeholder="example@gmail.com"
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="collectionDate">Blood Sample Collection Date *</Label>
                  <Input
                    id="collectionDate"
                    name="collectionDate"
                    type="date"
                    value={formData.collectionDate}
                    onChange={handleInputChange}
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="height">Height (cm) *</Label>
                  <Input
                    id="height"
                    name="height"
                    type="number"
                    value={formData.height}
                    onChange={handleInputChange}
                    placeholder="Height in cm"
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="weight">Weight (kg) *</Label>
                  <Input
                    id="weight"
                    name="weight"
                    type="number"
                    value={formData.weight}
                    onChange={handleInputChange}
                    placeholder="Weight in kg"
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="bloodPressure">Blood Pressure (mmHg) *</Label>
                  <Input
                    id="bloodPressure"
                    name="bloodPressure"
                    value={formData.bloodPressure}
                    onChange={handleInputChange}
                    placeholder="e.g., 120/80"
                  />
                </div>

                <div className="space-y-2 md:col-span-2 lg:col-span-3">
                  <Label htmlFor="medicalHistory">Medical History (Optional)</Label>
                  <Textarea
                    id="medicalHistory"
                    name="medicalHistory"
                    value={formData.medicalHistory}
                    onChange={handleInputChange}
                    placeholder="Any relevant medical history"
                    className="min-h-[100px]"
                  />
                </div>
              </div>

              <div className="flex flex-col sm:flex-row gap-4">
                <Button type="button" variant="outline" onClick={generatePatientId} disabled={isEditing}>
                  Generate Patient ID
                </Button>

                {formData.id && (
                  <div className="flex items-center gap-2 text-sm">
                    <span className="font-medium">Patient ID:</span>
                    <span className="bg-muted px-2 py-1 rounded">{formData.id}</span>
                  </div>
                )}

                <div className="flex-1"></div>

                <Button type="submit">{isEditing ? "Update Patient" : "Save Patient"}</Button>

                {isEditing && (
                  <Button type="button" variant="outline" onClick={resetForm}>
                    Cancel
                  </Button>
                )}
              </div>
            </form>
          </CardContent>
        </Card>

        {/* Last Added Patient Card */}
        {lastAddedPatient && (
          <Card className="mb-8 border-2 border-primary/20">
            <CardContent className="pt-6 pb-6">
              <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
                <div>
                  <h3 className="text-lg font-semibold mb-2">Recently Added Patient</h3>
                  <p className="text-sm text-muted-foreground mb-2">
                    Patient ID: <span className="font-medium">{lastAddedPatient.id}</span>
                  </p>
                  <p className="text-sm mb-4">
                    <span className="font-medium">{lastAddedPatient.name}</span>, {lastAddedPatient.age} years,{" "}
                    {lastAddedPatient.gender}
                  </p>
                </div>
                <Button onClick={() => handleProceedToDetection(lastAddedPatient)} className="flex items-center gap-2">
                  <Droplet className="h-4 w-4" />
                  Proceed to Blood Detection
                </Button>
              </div>
            </CardContent>
          </Card>
        )}

        <div className="mb-4">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-xl font-bold">Patient Records</h2>
            <Button variant="outline" size="sm" onClick={handleRefreshData} className="flex items-center gap-2">
              <RefreshCw className="h-4 w-4" />
              Refresh Data
            </Button>
          </div>
          <div className="mb-4">
            <Input
              placeholder="Search by Patient ID or Name"
              value={filterValue}
              onChange={(e) => setFilterValue(e.target.value)}
              className="max-w-sm"
            />
          </div>

          <div className="overflow-x-auto">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Patient ID</TableHead>
                  <TableHead>Name</TableHead>
                  <TableHead>Age</TableHead>
                  <TableHead>Gender</TableHead>
                  <TableHead>Gmail</TableHead>
                  <TableHead>Collection Date</TableHead>
                  <TableHead>Blood Group</TableHead>
                  <TableHead>Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {filteredPatients.length > 0 ? (
                  filteredPatients.map((patient) => (
                    <TableRow key={patient.id}>
                      <TableCell className="font-medium">{patient.id}</TableCell>
                      <TableCell>{patient.name}</TableCell>
                      <TableCell>{patient.age}</TableCell>
                      <TableCell>{patient.gender}</TableCell>
                      <TableCell>{patient.email}</TableCell>
                      <TableCell>{patient.collectionDate}</TableCell>
                      <TableCell>
                        <span
                          className={
                            patient.detectionHistory === "Not Tested"
                              ? "text-muted-foreground"
                              : "font-medium text-primary"
                          }
                        >
                          {patient.detectionHistory}
                        </span>
                      </TableCell>
                      <TableCell>
                        <div className="flex items-center gap-2">
                          <Button variant="ghost" size="icon" onClick={() => handleEdit(patient)}>
                            <Edit className="h-4 w-4" />
                            <span className="sr-only">Edit</span>
                          </Button>
                          <Button variant="ghost" size="icon" onClick={() => handleProceedToDetection(patient)}>
                            <Droplet className="h-4 w-4" />
                            <span className="sr-only">Blood Detection</span>
                          </Button>
                          {patient.detectionHistory !== "Not Tested" && (
                            <Button variant="ghost" size="icon" onClick={() => handleViewReport(patient)}>
                              <FileText className="h-4 w-4" />
                              <span className="sr-only">View Report</span>
                            </Button>
                          )}
                          <Button variant="ghost" size="icon" onClick={() => handleDelete(patient.id)}>
                            <Trash2 className="h-4 w-4" />
                            <span className="sr-only">Delete</span>
                          </Button>
                        </div>
                      </TableCell>
                    </TableRow>
                  ))
                ) : (
                  <TableRow>
                    <TableCell colSpan={8} className="text-center py-4">
                      No patients found
                    </TableCell>
                  </TableRow>
                )}
              </TableBody>
            </Table>
          </div>
        </div>
      </main>

      <AlertDialog open={deleteDialogOpen} onOpenChange={setDeleteDialogOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Are you sure?</AlertDialogTitle>
            <AlertDialogDescription>
              This action cannot be undone. This will permanently delete the patient record.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction onClick={confirmDelete}>Delete</AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  )
}